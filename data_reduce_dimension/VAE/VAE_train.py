#coding:utf-8


from tkinter.messagebox import NO
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from VAE import VAE, loss_function
import torch, gc
import numpy as np
import tqdm
import sys
sys.path.append('../../my_dataset')
from my_dataset import My_Dataset, get_samples

# # 半精度训练
# from torch.cuda.amp import autocast

"""
训练 VAE 模型

"""

def set_seed(seed:int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    
def main(epochs=100, learning_rate=1e-3, device=torch.device('cpu'), 
        train_sample_path='my_dataset/train.txt', val_sample_path='my_dataset/validation.txt',
        batch_size=5):

        # 0 创建数据集
        train_samples = get_samples(train_sample_path)
        validation_samples = get_samples(val_sample_path)
        train_dataset = My_Dataset('../../data/train', train_samples)
        val_dataset = My_Dataset('../../data/validation', validation_samples)

        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 1 建立模型
        module_1 = VAE(396065, [800, 400, 200, 128]) # dna
        # module_2 = VAE(18574, [512, 256, 128, 96])  # rna
        # module_3 = VAE(217, [200,100, 60, 32])   # rppa
        module_1 = module_1.to(device)
        # module_2 = module_2.to(device)
        # module_3 = module_3.to(device)

        # 2 构建损失函数 loss_function
        loss_1 = loss_function
        # loss_2 = loss_function
        # loss_3 = loss_function    
        # 3 构建优化器 optimizer
        optimizer_1 = torch.optim.Adam(module_1.parameters(), lr=learning_rate)
        # optimizer_2 = torch.optim.Adam(module_2.parameters(), lr=learning_rate)
        # optimizer_3 = torch.optim.Adam(module_3.parameters(), lr=learning_rate)
        # 4 设置超参数 epochs, learning_rate
        # 5 设置其它记录变量
        train_loss_1, train_loss_2, train_loss_3 = [], [], []
        val_loss_1, val_loss_2, val_loss_3 = [], [], []

        # 6 开始训练
        for epoch in range(epochs):
            module_1.train()
            # module_2.train()
            # module_3.train()

            train_loss_sum1, train_loss_sum2, train_loss_sum3 = 0, 0, 0
            for x1, x2, x3, y in train_data:
                x1 = x1.to(torch.float32).to(device)
                # x2 = x2.to(torch.float32).to(device)
                # x3 = x3.to(torch.float32).to(device)
                y = y.to(torch.int).to(device)

                # 前向传播
                z1, recon_x1, mean1, log_var1 = module_1(x1)
                # z2, recon_x2, mean2, log_var2 = module_2(x2)
                # z3, recon_x3, mean3, log_var3 = module_3(x3)


                # 梯度清0
                optimizer_1.zero_grad()
                # optimizer_2.zero_grad()
                # optimizer_3.zero_grad()

                # 计算损失函数
                temp_ls_1 = loss_1(x1, recon_x1, mean1, log_var1)
                # temp_ls_2 = loss_2(x2, recon_x2, mean2, log_var2)
                # temp_ls_3 = loss_3(x3, recon_x3, mean3, log_var3)

                train_loss_sum1 += temp_ls_1.sum().item()
                # train_loss_sum2 += temp_ls_2.sum().item()
                # train_loss_sum3 += temp_ls_3.sum().item()


                # 计算反向传播
                temp_ls_1.backward()
                # temp_ls_2.backward()
                # temp_ls_3.backward()

                # 优化  
                optimizer_1.step()
                # optimizer_2.step()
                # optimizer_3.step()

            print("Epoch: %d | Train loss: %.4f (DNA)" % (epoch+1, train_loss_sum1))
            # print("Epoch: %d | Train loss:  %.4f (RNA)" % (epoch+1, train_loss_sum2))  
            # print("Epoch: %d | Train loss:  %.4f (RPPA)" % (epoch+1, train_loss_sum3))              
            train_loss_1.append(train_loss_sum1)
            # train_loss_2.append(train_loss_sum2)
            # train_loss_3.append(train_loss_sum3)

            # 模型评估
            module_1.eval()
            # module_2.eval()
            # module_3.eval()

            with torch.no_grad():
                val_loss_sum1, val_loss_sum2, val_loss_sum3 = 0, 0, 0
                for x1, x2, x3, y in val_data:
                    x1 = x1.to(torch.float32).to(device)
                    x2 = x2.to(torch.float32).to(device)
                    x3 = x3.to(torch.float32).to(device)
                    y = y.to(torch.int).to(device)

                    z1, recon_x1, mean1, log_var1 = module_1(x1)
                    # z2, recon_x2, mean2, log_var2 = module_2(x2)
                    # z3, recon_x3, mean3, log_var3 = module_3(x3)

                    temp_ls_1 = loss_1(x1, recon_x1, mean1, log_var1)
                    # temp_ls_2 = loss_2(x2, recon_x2, mean2, log_var2)
                    # temp_ls_3 = loss_3(x3, recon_x3, mean3, log_var3)

                    val_loss_sum1 += temp_ls_1.sum().item()
                    # val_loss_sum2 += temp_ls_2.sum().item()
                    # val_loss_sum3 += temp_ls_3.sum().item()

            print("Epoch: %d | Validation loss: %.4f (DNA)" % 
                 (epoch+1, val_loss_sum1))
            # print("Epoch: %d | Validation loss:  %.4f (RNA)" % 
            #      (epoch+1, val_loss_sum2))  
            # print("Epoch: %d | Validation loss:  %.4f (RPPA)" % 
            #      (epoch+1, val_loss_sum3))                         
            val_loss_1.append(val_loss_sum1)
            # val_loss_2.append(val_loss_sum2)
            # val_loss_3.append(val_loss_sum3)

            if((epoch+1) % 10 == 0):
                torch.save(module_1, 'result/module/dna/module_1_{}.pkl'.format(epoch+1))
                # torch.save(module_2, 'result/module/rna/module_2_{}.pkl'.format(epoch+1))
                # torch.save(module_3, 'result/module/rppa/module_3_{}.pkl'.format(epoch+1))
            

        # 画图
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.plot([i for i in range(len(train_loss_1))], train_loss_1, '-', [i for i in range(len(val_loss_1))], val_loss_1, '--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('DNA')
        plt.legend()

        # plt.subplot(1, 3, 2)
        # plt.plot([i for i in range(len(train_loss_2))], train_loss_2, '-', [i for i in range(len(val_loss_2))], val_loss_2, '--')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('RNA')
        # plt.legend()

        # plt.subplot(1, 3, 3)
        # plt.plot([i for i in range(len(train_loss_3))], train_loss_3, '-', [i for i in range(len(val_loss_3))], val_loss_3, '--')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('RPPA')
        # plt.legend()

        plt.savefig('result/Train loss.png')
        plt.show()

def extract_features(fea1_path:str, fea2_path:str, fea3_path:str, topn:int=100) -> None:
    df1 = pd.read_csv(fea1_path, index_col=0, iterator=True, chunksize=1)  # DNA
    df2 = pd.read_csv(fea2_path, index_col=0, iterator=True, chunksize=1)  # RNA
    df3 = pd.read_csv(fea3_path, index_col=0, iterator=True, chunksize=1)  # RPPA
    for dna, rna, rppa in zip(df1, df2, df3):
        fea1_omics = dna.columns.tolist()
        fea2_omics = rna.columns.tolist()
        fea3_omics = rppa.columns.tolist()
        break

    topn_omics1, topn_omics2, topn_omics3 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in range(9, 100, 10):
        model1_path = 'result/module/dna/module_1_{}.pkl'.format(i+1)
        model2_path = 'result/module/rna/module_2_{}.pkl'.format(i+1)
        model3_path = 'result/module/rppa/module_3_{}.pkl'.format(i+1)

        model_1 = torch.load(model1_path)
        model_2 = torch.load(model2_path)
        model_3 = torch.load(model3_path)

        model_1_dict = model_1.state_dict()
        model_2_dict = model_2.state_dict()
        model_3_dict = model_3.state_dict()

        weight_omics1 = np.abs(model_1_dict['encoder.0.weight'].detach().cpu().numpy().T)
        weight_omics2 = np.abs(model_2_dict['encoder.0.weight'].detach().cpu().numpy().T)
        weight_omics3 = np.abs(model_3_dict['encoder.0.weight'].detach().cpu().numpy().T)

        weight_omics1_df = pd.DataFrame(weight_omics1, index=fea1_omics)
        weight_omics2_df = pd.DataFrame(weight_omics2, index=fea2_omics)
        weight_omics3_df = pd.DataFrame(weight_omics3, index=fea3_omics)

        weight_omics1_df['Importance'] = weight_omics1_df.apply(lambda x: x.sum(), axis=1)
        weight_omics2_df['Importance'] = weight_omics2_df.apply(lambda x: x.sum(), axis=1)
        weight_omics3_df['Importance'] = weight_omics3_df.apply(lambda x: x.sum(), axis=1)

        fea1_omics_topn = weight_omics1_df.nlargest(topn, 'Importance').index.tolist()
        fea2_omics_topn = weight_omics2_df.nlargest(topn, 'Importance').index.tolist()
        fea3_omics_topn = weight_omics3_df.nlargest(topn, 'Importance').index.tolist()

        omics1_col_name, omics2_col_name, omics3_col_name = str(i+1), str(i+1), str(i+1)

        topn_omics1[omics1_col_name] = fea1_omics_topn
        topn_omics2[omics2_col_name] = fea2_omics_topn
        topn_omics3[omics3_col_name] = fea3_omics_topn

    save1_path = 'result/topn_biomarker/dna_topn.csv'
    save2_path = 'result/topn_biomarker/rna_topn.csv'
    save3_path = 'result/topn_biomarker/rppa_topn.csv'
    topn_omics1.to_csv(save1_path, header=True, index=False)
    topn_omics2.to_csv(save2_path, header=True, index=False)
    topn_omics3.to_csv(save3_path, header=True, index=False)





import os

if __name__ == "__main__":
    set_seed(0)
    print("Starting to train VAE ......")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    main(epochs=100, learning_rate=1e-3, device=torch.device('cuda'), 
        train_sample_path='../../my_dataset/train.txt', val_sample_path='../../my_dataset/validation.txt',
        batch_size=2)
    print("Train successful, please see results from the file result.")
