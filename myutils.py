# coding: utf-8

from data_reduce_dimension.VAE.VAE import RPPA_VAE, RNA_VAE, DNA_VAE, segment_x, rppa_rna_loss, dna_loss
from data_fusion.PFA.PFA_main import PFA_main
from classification.GTCN.GTNet.GNN_models import GTCN
from classification.GTCN.GTNet.utils import get_GTCN_g
from my_dataset.my_dataset import load_data

import torch
from torch import nn 
import argparse
import matplotlib.pyplot as plt
from math import inf
import time
import warnings
warnings.filterwarnings('ignore')

class End2end(nn.Module):
    def __init__(self, opt:argparse.Namespace, device) -> None:
        super(End2end, self).__init__()
        self.opt = opt
        self.device = device
        # DNA
        self.layer1 = DNA_VAE(opt.DNA_IN, mode=opt.MODE)
        # RNA
        self.layer2 = RNA_VAE(opt.RNA_IN, mode=opt.MODE)
        # RPPA
        self.layer3 = RPPA_VAE(opt.RPPA_IN, mode=opt.MODE)
        # GTCN
        self.layer4 = GTCN(n_in=320, n_hid=128, n_out=33, dropout=0.5, dropout2=0.25, hop=opt.HOP)

    def forward(self, x1, x2, x3):
        mean1, log_var1, recon_x1, z1 = self.layer1(x1)
        mean2, log_var2, recon_x2, z2 = self.layer2(x2)
        mean3, log_var3, recon_x3, z3 = self.layer3(x3)

        mean = torch.cat([mean1, mean2, mean3], axis=1)
        adj = PFA_main(mean1, mean2, mean3, self.opt.THRESHOLD, self.device)
        g = get_GTCN_g(adj, self.device)

        output = self.layer4(mean, g)

        return output 



def twostage(opt:argparse.Namespace):
    # '2stage' mode
    device = torch.device(opt.DEVICE)
    # 1. CONSTRUCT MODEL    
    models = []
    models.append(DNA_VAE(opt.DNA_IN, mode=opt.MODE))
    models.append(RNA_VAE(opt.RNA_IN, mode=opt.MODE))
    models.append(RPPA_VAE(opt.RPPA_IN, mode=opt.MODE))
    # models.append(GTCN(n_in=320, n_hid=128, n_out=33, dropout=0.5, dropout2=0.25, hop=opt.HOP))

    model_num = len(models)
    if (opt.DEVICE == 'cuda'):
        for idx in range(model_num):
            models[idx] = torch.nn.DataParallel(models[idx])

    for idx in range(model_num):
        models[idx] = models[idx].to(device)

    # 2. CONSTRUCT LOSS FUNCTION
    models_loss = []
    models_loss.append(dna_loss)
    models_loss.append(rppa_rna_loss)
    models_loss.append(rppa_rna_loss)

    # 3. CONSTRUCT OPTIMIZER

    # 4. LOAD DATALOADER
    train_dataloader, val_dataloader = load_data(opt.BATCHSIZE)

    # 5. SET HYPERPARAMETER
    lr = [opt.LR1, opt.LR2, opt.LR3]

    print('First Stage: Training VAE models for DNA, RNA and RPPA data.')
    for index, model in enumerate(models):
        print('Start to train model {}......'.format(index+1))
        optimizer = torch.optim.Adam(model.parameters(), lr[index])

        TRAIN_ACC, VAL_ACC = [], []
        TRAIN_LOSS, VAL_LOSS = [], []
        TO_SAVE_MODEL = inf

        # 6. START TRAINING
        for epoch in range(opt.EPOCHS):
            print('-'*50, 'EPOCH: ', epoch+1, '-'*50)
            model.train()
            loss_sum = 0.0
            acc = 0.0
            for x1, x2, x3, y in train_dataloader:
                if (index == 0):
                    x1 = x1.to(torch.float32).to(device)
                    x = segment_x(x1, opt.CHR_PATH)
                elif(index == 1):
                    x = x2.to(torch.float32).to(device)
                elif(index == 2):
                    x = x3.to(torch.float32).to(device)

                y = y.to(torch.float32).to(device)    
                mean, log_var, z, recon_x, pre_y = model(x)

                optimizer.zero_grad()
                temp = models_loss[index](recon_x, x, mean, log_var, pre_y, y, mode=opt.MODE)
                loss_sum += temp.item()
                acc += (torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)).item()
                temp.backward()
                optimizer.step()

            train_acc1_temp = acc / len(train_dataloader)
            train_loss1_temp = loss_sum / len(train_dataloader)
            TRAIN_ACC.append(train_acc1_temp)
            TRAIN_LOSS.append(train_loss1_temp)

            model.eval()
            with torch.no_grad():
                loss_sum = 0.0
                acc = 0.0
                for x1, x2, x3, y in val_dataloader:
                    if(index == 0):
                        x1 = x1.to(torch.float32).to(device)
                        x = segment_x(x1, opt.CHR_PATH)
                    elif(index == 1):
                        x = x2.to(torch.float32).to(device)        
                    elif(index == 2):
                        x = x3.to(torch.float32).to(device)

                    y = y.to(torch.float32).to(device)     
                    mean, log_var, z, recon_x, pre_y = model(x)

                    temp = models_loss[index](recon_x, x, mean, log_var, pre_y, y, mode=opt.MODE)

                    loss_sum += temp.item()
                    acc += (torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)).item()
 

            val_acc1_temp = acc / len(val_dataloader)
            val_loss1_temp = loss_sum / len(val_dataloader)
            VAL_ACC.append(val_acc1_temp)
            VAL_LOSS.append(val_loss1_temp)


            if(VAL_LOSS[-1] < TO_SAVE_MODEL):
                TO_SAVE_MODEL= VAL_LOSS[-1]
                torch.save(model, 'result/2stage/model{}.pkl'.format(index+1))  

            print(f'Model_{index+1}: train loss: {TRAIN_LOSS[-1]}, val loss: {VAL_LOSS[-1]}, train acc: {TRAIN_ACC[-1]*100}, val acc: {VAL_ACC[-1]*100} \
                 ')

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(list(range(len(TRAIN_LOSS))), TRAIN_LOSS, '-', label='train')
        plt.plot(list(range(len(VAL_LOSS))), VAL_LOSS, '--', label='validation')
        plt.title('MODEL {} Loss'.format(index+1))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(list(range(len(TRAIN_ACC))), TRAIN_ACC, '-', label='train')
        plt.plot(list(range(len(VAL_ACC))), VAL_ACC, '--', label='validation')
        plt.title('MODEL {} Accrucay'.format(index+1))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('result/2stage/MODEL{}_Train.png'.format(index+1))

    time.sleep(5)
    print('Second Stage: Training GTCN model for tumor classification.')

    model1 = torch.load('result/2stage/model1.pkl')
    model2 = torch.load('result/2stage/model2.pkl')
    model3 = torch.load('result/2stage/model3.pkl')
    model = GTCN(n_in=320, n_hid=128, n_out=33, dropout=0.5, dropout2=0.25, hop=opt.HOP)

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), opt.LR4)

    TRAIN_LOSS, VAL_LOSS = [], []
    TRAIN_ACC, VAL_ACC = [], []
    TO_SAVE_MODEL = inf

    for epoch in range(opt.EPOCHS):
        print('-'*50, 'EPOCH:', epoch+1, '-'*50)

        model.train()
        loss_sum, acc_sum = 0.0, 0.0
        for x1, x2, x3, y in train_dataloader:
            x1 = x1.to(torch.float32).to(device)
            x1 = segment_x(x1, opt.CHR_PATH)
            x2 = x2.to(torch.float32).to(device)
            x3 = x3.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device) 

            mean1, log_var1, recon_x1, z1, y_pre1 = model1(x1)   # dna
            mean2, log_var2, recon_x2, z2, y_pre2 = model2(x2)   # rna
            mean3, log_var3, recon_x3, z3, y_pre3 = model3(x3)   # rppa   

            mean = torch.cat([mean1, mean2, mean3], axis=1)      # used for inputs to GTCN
            adj = PFA_main(mean1, mean2, mean3, threshold=opt.THRESHOLD, device=device)
            g = get_GTCN_g(adj, device=device)

            pre_y = model(mean, g)

            optimizer.zero_grad()

            loss = loss_fn(torch.squeeze(pre_y), y.long())

            loss_sum += loss.item()
            acc_temp = torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)
            acc_sum += acc_temp.item()

            loss.backward()
            optimizer.step()
        
        TRAIN_LOSS.append(loss_sum/len(train_dataloader))
        TRAIN_ACC.append(acc_sum/len(train_dataloader))

        model.eval()
        loss_sum, acc_sum = 0.0, 0.0
        with torch.no_grad():
            for x1, x2, x3, y in val_dataloader:
                x1 = x1.to(torch.float32).to(device)
                x1 = segment_x(x1, opt.CHR_PATH)
                x2 = x2.to(torch.float32).to(device)
                x3 = x3.to(torch.float32).to(device)
                y = y.to(torch.float32).to(device) 

                mean1, log_var1, recon_x1, z1, y_pre1 = model1(x1)   # dna
                mean2, log_var2, recon_x2, z2, y_pre2 = model2(x2)   # rna
                mean3, log_var3, recon_x3, z3, y_pre3 = model3(x3)   # rppa   

                mean = torch.cat([mean1, mean2, mean3], axis=1)      # used for inputs to GTCN
                adj = PFA_main(mean1, mean2, mean3, threshold=opt.THRESHOLD, device=device)
                g = get_GTCN_g(adj, device=device)

                pre_y = model(mean, g)

                loss_sum += loss.item()
                acc_temp = torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)
                acc_sum += acc_temp.cpu().item()
        
        VAL_LOSS.append(loss_sum/len(val_dataloader))
        VAL_ACC.append(acc_sum/len(val_dataloader))
        if(VAL_LOSS[-1] < TO_SAVE_MODEL):
            TO_SAVE_MODEL[3] = VAL_LOSS[-1]
            torch.save(model, 'result/2stage/model4.pkl')

        print(f'GTCN-----Training loss: {TRAIN_LOSS[-1]}, Val loss: {VAL_LOSS[-1]} \n \
---------Training acc: {TRAIN_ACC[-1]*100} %, Val acc: {VAL_ACC[-1]*100} %.')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(len(TRAIN_LOSS))), TRAIN_LOSS, '-', label='train')
    plt.plot(list(range(len(VAL_LOSS))), VAL_LOSS, '--', label='validation')
    plt.title('GTCN loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(TRAIN_ACC))), TRAIN_ACC, '-', label='train')
    plt.plot(list(range(len(VAL_ACC))), VAL_ACC, '--', label='validation')
    plt.title('GTCN accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('result/2stage/GTCN_train.png')
    # 7. RECORD AND STORE RESULT 
    print('Training successfully. See result from the result file.')



# Old version, dropped. 
def mode_one(opt:argparse.Namespace):
    # '2stage' mode
    device = torch.device(opt.DEVICE)
    # 1. CONSTRUCT MODEL
    model1 = DNA_VAE(opt.DNA_IN, mode=opt.MODE)
    model2 = RNA_VAE(opt.RNA_IN, mode=opt.MODE)
    model3 = RPPA_VAE(opt.RPPA_IN, mode=opt.MODE)
    model4 = GTCN(n_in=320, n_hid=128, n_out=33, dropout=0.5, dropout2=0.25, hop=opt.HOP)

    if (opt.DEVICE == 'cuda'):
        model1 = torch.nn.DataParallel(model1)
        model2 = torch.nn.DataParallel(model2)
        model3 = torch.nn.DataParallel(model3)
        model4 = torch.nn.DataParallel(model4)
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model4 = model4.to(device)

    # 2. CONSTRUCT LOSS FUNCTION
    md1_ls = dna_loss
    md2_ls = rppa_rna_loss
    md3_ls = rppa_rna_loss
    md4_ls = nn.CrossEntropyLoss(reduction='sum')

    # 3. CONSTRUCT OPTIMIZER
    optimizer1 = torch.optim.Adam(model1.parameters(), 1e-3)
    optimizer2 = torch.optim.Adam(model2.parameters(), 1e-3)
    optimizer3 = torch.optim.Adam(model3.parameters(), 1e-6)
    optimizer4 = torch.optim.Adam(model4.parameters(), opt.LR4)

    # 4. LOAD DATALOADER
    train_dataloader, val_dataloader = load_data(opt.BATCHSIZE)

    # 5. SET HYPERPARAMETER
    TRAIN_ACC1, VAL_ACC1 = [], []
    TRAIN_ACC2, VAL_ACC2 = [], []
    TRAIN_ACC3, VAL_ACC3 = [], []
    TRAIN_ACC4, VAL_ACC4 = [], []
    TRAIN_LOSS1, VAL_LOSS1 = [], []
    TRAIN_LOSS2, VAL_LOSS2 = [], []
    TRAIN_LOSS3, VAL_LOSS3 = [], []
    TRAIN_LOSS4, VAL_LOSS4 = [], []
    TO_SAVE_MODEL = [inf, inf, inf, inf]  # model1 .... model4. 

    # 6. START TRAINING
    print('First Stage: Training VAE models for DNA, RNA and RPPA data.')
    for epoch in range(opt.EPOCHS):
        print('-'*50, 'EPOCH: ', epoch+1, '-'*50)
        model1.train()
        model2.train()
        model3.train()
        loss_sum1, loss_sum2, loss_sum3 = 0.0, 0.0, 0.0
        acc1, acc2, acc3 = 0.0, 0.0, 0.0
        for x1, x2, x3, y in train_dataloader:
            x1 = x1.to(torch.float32).to(device)
            x1 = segment_x(x1, opt.CHR_PATH)
            x2 = x2.to(torch.float32).to(device)
            x3 = x3.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            mean1, log_var1, z1, recon_x1, pre_y1 = model1(x1)
            mean2, log_var2, z2, recon_x2, pre_y2 = model2(x2)
            mean3, log_var3, z3, recon_x3, pre_y3 = model3(x3)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            temp1 = md1_ls(recon_x1, x1, mean1, log_var1, pre_y1, y, mode=opt.MODE)
            temp2 = md2_ls(recon_x2, x2, mean2, log_var2, pre_y2, y, opt.MODE)
            temp3 = md3_ls(recon_x3, x3, mean3, log_var3, pre_y3, y, opt.MODE)

            loss_sum1 += temp1.item()
            loss_sum2 += temp2.item()
            loss_sum3 += temp3.item()
            acc1 += (torch.sum(torch.argmax(pre_y1, axis=1) == y) / len(pre_y1)).item()
            acc2 += (torch.sum(torch.argmax(pre_y2, axis=1) == y) / len(pre_y2)).item()
            acc3 += (torch.sum(torch.argmax(pre_y3, axis=1) == y) / len(pre_y3)).item()

            temp1.backward()
            temp2.backward()
            temp3.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

        train_acc1_temp = acc1 / len(train_dataloader)
        train_acc2_temp = acc2 / len(train_dataloader)
        train_acc3_temp = acc3 / len(train_dataloader)
        train_loss1_temp = loss_sum1 / len(train_dataloader)
        train_loss2_temp = loss_sum2 / len(train_dataloader)
        train_loss3_temp = loss_sum3 / len(train_dataloader)
        TRAIN_ACC1.append(train_acc1_temp)
        TRAIN_ACC2.append(train_acc2_temp)
        TRAIN_ACC3.append(train_acc3_temp)
        TRAIN_LOSS1.append(train_loss1_temp)
        TRAIN_LOSS2.append(train_loss2_temp)
        TRAIN_LOSS3.append(train_loss3_temp)

        model1.eval()
        model2.eval()
        model3.eval()
        with torch.no_grad():
            loss_sum1, loss_sum2, loss_sum3 = 0.0, 0.0, 0.0
            acc1, acc2, acc3 = 0.0, 0.0, 0.0
            for x1, x2, x3, y in val_dataloader:
                x1 = x1.to(torch.float32).to(device)
                x1 = segment_x(x1, opt.CHR_PATH)
                x2 = x2.to(torch.float32).to(device)
                x3 = x3.to(torch.float32).to(device)
                y = y.to(torch.float32).to(device)     

                mean1, log_var1, z1, recon_x1, pre_y1 = model1(x1)
                mean2, log_var2, z2, recon_x2, pre_y2 = model2(x2)
                mean3, log_var3, z3, recon_x3, pre_y3 = model3(x3)

                loss_sum1 += temp1.item()
                loss_sum2 += temp2.item()
                loss_sum3 += temp3.item()

                acc1 += (torch.sum(torch.argmax(pre_y1, axis=1) == y) / len(pre_y1)).item()
                acc2 += (torch.sum(torch.argmax(pre_y2, axis=1) == y) / len(pre_y2)).item()
                acc3 += (torch.sum(torch.argmax(pre_y3, axis=1) == y) / len(pre_y3)).item()   

        val_acc1_temp = acc1 / len(val_dataloader)
        val_acc2_temp = acc2 / len(val_dataloader)
        val_acc3_temp = acc3 / len(val_dataloader)
        val_loss1_temp = loss_sum1 / len(val_dataloader)
        val_loss2_temp = loss_sum2 / len(val_dataloader)
        val_loss3_temp = loss_sum3 / len(val_dataloader)
        VAL_ACC1.append(val_acc1_temp)
        VAL_ACC2.append(val_acc2_temp)
        VAL_ACC3.append(val_acc3_temp)
        VAL_LOSS1.append(val_loss1_temp)
        VAL_LOSS2.append(val_loss2_temp)
        VAL_LOSS3.append(val_loss3_temp)    

        if(VAL_LOSS1[-1] < TO_SAVE_MODEL[0]):
            TO_SAVE_MODEL[0] = VAL_LOSS1[-1]
            torch.save(model1, 'result/2stage/model1.pkl')  
        if(VAL_LOSS2[-1] < TO_SAVE_MODEL[1]):
            TO_SAVE_MODEL[1] = VAL_LOSS2[-1]
            torch.save(model2, 'result/2stage/model2.pkl')         
        if(VAL_LOSS3[-1] < TO_SAVE_MODEL[2]):
            TO_SAVE_MODEL[3] = VAL_LOSS3[-1]
            torch.save(model1, 'result/2stage/model3.pkl')  
        print(f'DNA: train loss: {TRAIN_LOSS1[-1]}, val loss: {VAL_LOSS1[-1]}, train acc: {TRAIN_ACC1[-1]*100}, val acc: {VAL_ACC1[-1]*100}\n \
                RNA: train loss: {TRAIN_LOSS2[-1]}, val loss: {VAL_LOSS2[-1]}, train acc: {TRAIN_ACC2[-1]*100}, val acc: {VAL_ACC2[-1]*100}\n \
                RPPA: train loss: {TRAIN_LOSS3[-1]}, val loss: {VAL_LOSS3[-1]}, train acc: {TRAIN_ACC3[-1]*100}, val acc: {VAL_ACC3[-1]*100}\n \
                ')

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    # DNA 
    plt.plot(list(range(len(TRAIN_LOSS1))), TRAIN_LOSS1, '-', label='train')
    plt.plot(list(range(len(VAL_LOSS1))), VAL_LOSS1, '--', label='validation')
    plt.title('VAE of DNA Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    # RNA
    plt.plot(list(range(len(TRAIN_LOSS2))), TRAIN_LOSS2, '-', label='train')
    plt.plot(list(range(len(VAL_LOSS2))), VAL_LOSS2, '--', label='validation')
    plt.title('VAE of RNA Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2, 3, 3)
    # RPPA
    plt.plot(list(range(len(TRAIN_LOSS3))), TRAIN_LOSS3, '-', label='train')
    plt.plot(list(range(len(VAL_LOSS3))), VAL_LOSS3, '--', label='validation')
    plt.title('VAE of RPPA Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(list(range(len(TRAIN_ACC1))), TRAIN_ACC1, '-', label='train')
    plt.plot(list(range(len(VAL_ACC1))), VAL_ACC1, '--', label='validation')
    plt.title('VAE of DNA Accrucay')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(list(range(len(TRAIN_ACC2))), TRAIN_ACC2, '-', label='train')
    plt.plot(list(range(len(VAL_ACC2))), VAL_ACC2, '--', label='validation')
    plt.title('VAE of RNA Accrucay')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(list(range(len(TRAIN_ACC3))), TRAIN_ACC3, '-', label='train')
    plt.plot(list(range(len(VAL_ACC3))), VAL_ACC3, '--', label='validation')
    plt.title('VAE of RPPA Accrucay')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig('result/2stage/VAE_Train.png')


    time.sleep(5)
    print('Second Stage: Training GTCN model for tumor classification.')

    model1 = torch.load('result/2stage/model1.pkl')
    model2 = torch.load('result/2stage/model2.pkl')
    model3 = torch.load('result/2stage/model3.pkl')

    for epoch in range(opt.EPOCHS):
        print('-'*50, 'EPOCH:', epoch+1, '-'*50)

        model4.train()
        loss_sum, acc_sum = 0.0, 0.0
        for x1, x2, x3, y in train_dataloader:
            x1 = x1.to(torch.float32).to(device)
            x1 = segment_x(x1, opt.CHR_PATH)
            x2 = x2.to(torch.float32).to(device)
            x3 = x3.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device) 

            mean1, log_var1, recon_x1, z1, y_pre1 = model1(x1)   # dna
            mean2, log_var2, recon_x2, z2, y_pre2 = model2(x2)   # rna
            mean3, log_var3, recon_x3, z3, y_pre3 = model3(x3)   # rppa   

            mean = torch.cat([mean1, mean2, mean3], axis=1)      # used for inputs to GTCN
            adj = PFA_main(mean1, mean2, mean3, threshold=opt.THRESHOLD, device=device)
            g = get_GTCN_g(adj, device=device)

            pre_y = model4(mean, g)

            optimizer4.zero_grad()

            loss = md4_ls(torch.squeeze(pre_y), y.long())

            loss_sum += loss.item()
            acc_temp = torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)
            acc_sum += acc_temp.item()

            loss.backward()
            optimizer4.step()
        
        TRAIN_LOSS4.append(loss_sum/len(train_dataloader))
        TRAIN_ACC4.append(acc_sum/len(train_dataloader))

        model4.eval()
        loss_sum, acc_sum = 0.0, 0.0
        with torch.no_grad():
            for x1, x2, x3, y in val_dataloader:
                x1 = x1.to(torch.float32).to(device)
                x1 = segment_x(x1, opt.CHR_PATH)
                x2 = x2.to(torch.float32).to(device)
                x3 = x3.to(torch.float32).to(device)
                y = y.to(torch.float32).to(device) 

                mean1, log_var1, recon_x1, z1, y_pre1 = model1(x1)   # dna
                mean2, log_var2, recon_x2, z2, y_pre2 = model2(x2)   # rna
                mean3, log_var3, recon_x3, z3, y_pre3 = model3(x3)   # rppa   

                mean = torch.cat([mean1, mean2, mean3], axis=1)      # used for inputs to GTCN
                adj = PFA_main(mean1, mean2, mean3, threshold=opt.THRESHOLD, device=device)
                g = get_GTCN_g(adj, device=device)

                pre_y = model4(mean, g)

                loss_sum += loss.item()
                acc_temp = torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)
                acc_sum += acc_temp.cpu().item()
        
        VAL_LOSS4.append(loss_sum/len(val_dataloader))
        VAL_ACC4.append(acc_sum/len(val_dataloader))
        if(VAL_LOSS4[-1] < TO_SAVE_MODEL[3]):
            TO_SAVE_MODEL[3] = VAL_LOSS4[-1]
            torch.save(model4, 'result/2stage/model4.pkl')

        print(f'GTCN-----Training loss: {TRAIN_LOSS4[-1]}, Val loss: {VAL_LOSS4[-1]} \n \
                ---------Training acc: {TRAIN_ACC4[-1]*100}, Val acc: {VAL_ACC4[-1]*100}')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(len(TRAIN_LOSS4))), TRAIN_LOSS4, '-', label='train')
    plt.plot(list(range(len(VAL_LOSS4))), VAL_LOSS4, '--', label='validation')
    plt.title('GTCN loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(TRAIN_ACC4))), TRAIN_ACC4, '-', label='train')
    plt.plot(list(range(len(VAL_ACC4))), VAL_ACC4, '--', label='validation')
    plt.title('GTCN accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig('result/2stage/GTCN_train.png')
    # 7. RECORD AND STORE RESULT 
    print('Training successfully. See result from the result file.')


def end2end(opt:argparse.Namespace):
    # 'end2end' mode
    device = torch.device(opt.DEVICE)

    # 1. CONSTRUCT MODEL
    model = End2end(opt, device)
    if (opt.DEVICE == 'cuda'):
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # 2. CONSTRUCT LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    # 3. CONSTRUCT OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), opt.LR)

    # 4. LOAD DATALOADER
    train_dataloader, val_dataloader = load_data(opt.BATCHSIZE)

    # 5. SET HYPERPARAMETER
    TRAIN_ACC, VAL_ACC = [], []
    TRAIN_LOSS, VAL_LOSS = [], []
    TO_SAVE_MODEL = inf

    # 6. START TRAINING
    print('END2END model training mode.')
    for epoch in range(opt.EPOCHS):
        print('-'*50, 'EPOCH:', epoch+1, '-'*50)
        model.train()
        loss_sum = 0.0
        acc_sum = 0.0
        for x1, x2, x3, y in train_dataloader:
            x1 = x1.to(torch.float32).to(device)
            x1 = segment_x(x1, opt.CHR_PATH)
            x2 = x2.to(torch.float32).to(device)
            x3 = x3.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)     

            pre_y = model(x1, x2, x3)

            optimizer.zero_grad()

            loss = loss_fn(torch.squeeze(pre_y), y.long())

            loss_sum += loss.item()
            acc_temp = torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)
            acc_sum += acc_temp.item()

            loss.backward()
            optimizer.step()
        
        TRAIN_LOSS.append(loss_sum/len(train_dataloader))
        TRAIN_ACC.append(acc_sum/len(train_dataloader))

        model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            acc_sum = 0.0
            for x1, x2, x3, y in val_dataloader:
                x1 = x1.to(torch.float32).to(device)
                x1 = segment_x(x1, opt.CHR_PATH)
                x2 = x2.to(torch.float32).to(device)
                x3 = x3.to(torch.float32).to(device)
                y = y.to(torch.float32).to(device)     

                pre_y = model(x1, x2, x3)

                loss = loss_fn(torch.squeeze(pre_y), y.long())

                loss_sum += loss.item()
                acc_temp = torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)
                acc_sum += acc_temp.item()                

        VAL_LOSS.append(loss_sum/len(val_dataloader))
        VAL_ACC.append(acc_sum/len(val_dataloader))

        if(VAL_LOSS[-1] < TO_SAVE_MODEL):
            TO_SAVE_MODEL = VAL_LOSS[-1]
            torch.save(model, 'result/end2end/end2end_model.pkl')

        print(f'Omi-PGTCN (End2end model): Training loss: {TRAIN_LOSS[-1]}, Val loss: {VAL_LOSS[-1]}\n \
-----------------Training acc: {TRAIN_ACC[-1]*100} %, Val acc: {VAL_ACC[-1]*100 } %.')
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(len(TRAIN_LOSS))), TRAIN_LOSS, '-', label='train')
    plt.plot(list(range(len(VAL_LOSS))), VAL_LOSS, '--', label='validation')
    plt.title('Omi-PGTCN (end2end) loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(TRAIN_ACC))), TRAIN_ACC, '-', label='train')
    plt.plot(list(range(len(VAL_ACC))), VAL_ACC, '--', label='validation')
    plt.title('Omi-PGTCN (end2end) accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('result/end2end/OmiPGTCN_train.png')
    print('Training successfully. See result from the result file.')

