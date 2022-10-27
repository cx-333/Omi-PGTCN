#coding:utf-8

import torch.nn.functional as F
import torch 
from torch import nn, Tensor
# Half precision training 
# from torch.cuda.amp import autocast


# dimensionality reduction networck for protein expression
class RPPA_VAE(nn.Module):
    def __init__(self, input_dim:int=217, mode:str='2stage') -> None:
        super(RPPA_VAE, self).__init__()
        self.mode = mode
        # encoder 
        self.encode = self.fc_layer(input_dim, 128)

        self.mean = self.fc_layer(128, 64, activation=0)
        self.log_var = self.fc_layer(128, 64, activation=0)

        # decoder
        self.decode_1 = self.fc_layer(64, 128)
        self.decode_2 = self.fc_layer(128, input_dim, activation=2)

        # classifier
        self.classify_1 = self.fc_layer(64, 33, activation=0)
        
        # Activation - 0: no activation,  1: ReLU,  2: Sigmoid
    def fc_layer(self, in_dim:int, out_dim:int, activation:int=1, dropout:bool=False, dropout_p=0.5):
        if activation == 0:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        elif activation == 2:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Sigmoid())
        else:
            if dropout:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
            else:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())
        return layer        

    def encoder(self, x):
        temp = self.encode(x)
        mean = self.mean(temp)
        log_var = self.log_var(temp)
        return mean, log_var

    def reparameterization(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decoder(self, z):
        temp = self.decode_1(z)
        out = self.decode_2(temp)
        return out
   
    # Half precision training, uncomment before using it.
    # @autocast()   
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        recon_x = self.decoder(z)
        if(self.mode == '2stage'):
            pre_y = self.classify_1(mean)
            return mean, log_var, z, recon_x, pre_y
        else:
            return mean, log_var, z, recon_x

# dimensionality reduction network for RNA expression
class RNA_VAE(nn.Module):
    def __init__(self, input_dim:int, latent_dim:list=[4096, 1028, 512, 128],
                 classifier_dim:list=[128, 64, 33], mode:str='2stage') -> None:
        """
        params:
        latent_dim: a list of which length is 4.
        classifier_dim:  a list of which length is 3.
        """
        super(RNA_VAE, self).__init__()
        self.mode = mode
        # Encoder
        self.e_fc1_expr = self.fc_layer(input_dim, latent_dim[0])
        self.e_fc2_expr = self.fc_layer(latent_dim[0], latent_dim[1])
        self.e_fc3 = self.fc_layer(latent_dim[1], latent_dim[2])
        self.e_fc4_mean = self.fc_layer(latent_dim[2], latent_dim[3], activation=0)
        self.e_fc4_log_var = self.fc_layer(latent_dim[2], latent_dim[3], activation=0)

        # Decoder
        self.d_fc4 = self.fc_layer(latent_dim[3], latent_dim[2])
        self.d_fc3 = self.fc_layer(latent_dim[2], latent_dim[1])
        self.d_fc2_expr = self.fc_layer(latent_dim[1], latent_dim[0])
        self.d_fc1_expr = self.fc_layer(latent_dim[0], input_dim, activation=2)

        # Classifier
        self.c_fc1 = self.fc_layer(latent_dim[3], classifier_dim[0])
        self.c_fc2 = self.fc_layer(classifier_dim[0], classifier_dim[1])
        self.c_fc3 = self.fc_layer(classifier_dim[1], classifier_dim[2], activation=0)

        # Activation - 0: no activation,  1: ReLU,  2: Sigmoid
    def fc_layer(self, in_dim, out_dim, activation=1, dropout=False, dropout_p=0.5):
        if activation == 0:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim))
        elif activation == 2:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Sigmoid())
        else:
            if dropout:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
            else:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())
        return layer

    def encode(self, x):
        expr_level2_layer = self.e_fc1_expr(x)
        level3_layer = self.e_fc2_expr(expr_level2_layer)
        level4_layer = self.e_fc3(level3_layer)
        latent_mean = self.e_fc4_mean(level4_layer)
        latent_log_var = self.e_fc4_log_var(level4_layer)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decode(self, z):
        level_4_layer = self.d_fc4(z)
        level_3_layer = self.d_fc3(level_4_layer)
        expr_level2_layer = self.d_fc2_expr(level_3_layer)
        recon_x = self.d_fc1_expr(expr_level2_layer)
        return recon_x    
    
    def classifier(self, mean):
        level_1_layer = self.c_fc1(mean)
        level_2_layer = self.c_fc2(level_1_layer)
        output_layer = self.c_fc3(level_2_layer)
        return output_layer

    # Half precision training, uncomment before using it.
    # @autocast() 
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        if(self.mode == '2stage'):
            classifier_x = mean
            pred_y = self.classifier(classifier_x)
            return mean, log_var, z, recon_x, pred_y
        else:
            return mean, log_var, z, recon_x


# dimensionality reduction network for DNA methylation 
class DNA_VAE(nn.Module):
    def __init__(self, input_dim_methy_array:list, level_2_dim_methy:int=256,
                 level_3_dim_methy:int=1024, level_4_dim:int=512,
                 latent_space_dim:int=128, classifier_1_dim:int=128, 
                 classifier_2_dim:int=64,classifier_out_dim:int=33,
                 mode:str='2stage') -> None:
        super(DNA_VAE, self).__init__()
        self.level_2_dim_methy = level_2_dim_methy
        self.level_3_dim_methy = level_3_dim_methy
        self.level_4_dim = level_4_dim
        self.latent_space_dim = latent_space_dim
        self.classifier_1_dim = classifier_1_dim
        self.classifier_2_dim = classifier_2_dim
        self.classifier_out_dim = classifier_out_dim
        self.mode = mode
        # ENCODER fc layers
        # level 1
        # Methy input for each chromosome,  chr 1-22, X (0-22)
        self.e_fc1_methy_1 = self.fc_layer(input_dim_methy_array[0], level_2_dim_methy)
        self.e_fc1_methy_2 = self.fc_layer(input_dim_methy_array[1], level_2_dim_methy)
        self.e_fc1_methy_3 = self.fc_layer(input_dim_methy_array[2], level_2_dim_methy)
        self.e_fc1_methy_4 = self.fc_layer(input_dim_methy_array[3], level_2_dim_methy)
        self.e_fc1_methy_5 = self.fc_layer(input_dim_methy_array[4], level_2_dim_methy)
        self.e_fc1_methy_6 = self.fc_layer(input_dim_methy_array[5], level_2_dim_methy)
        self.e_fc1_methy_7 = self.fc_layer(input_dim_methy_array[6], level_2_dim_methy)
        self.e_fc1_methy_8 = self.fc_layer(input_dim_methy_array[7], level_2_dim_methy)
        self.e_fc1_methy_9 = self.fc_layer(input_dim_methy_array[8], level_2_dim_methy)
        self.e_fc1_methy_10 = self.fc_layer(input_dim_methy_array[9], level_2_dim_methy)
        self.e_fc1_methy_11 = self.fc_layer(input_dim_methy_array[10], level_2_dim_methy)
        self.e_fc1_methy_12 = self.fc_layer(input_dim_methy_array[11], level_2_dim_methy)
        self.e_fc1_methy_13 = self.fc_layer(input_dim_methy_array[12], level_2_dim_methy)
        self.e_fc1_methy_14 = self.fc_layer(input_dim_methy_array[13], level_2_dim_methy)
        self.e_fc1_methy_15 = self.fc_layer(input_dim_methy_array[14], level_2_dim_methy)
        self.e_fc1_methy_16 = self.fc_layer(input_dim_methy_array[15], level_2_dim_methy)
        self.e_fc1_methy_17 = self.fc_layer(input_dim_methy_array[16], level_2_dim_methy)
        self.e_fc1_methy_18 = self.fc_layer(input_dim_methy_array[17], level_2_dim_methy)
        self.e_fc1_methy_19 = self.fc_layer(input_dim_methy_array[18], level_2_dim_methy)
        self.e_fc1_methy_20 = self.fc_layer(input_dim_methy_array[19], level_2_dim_methy)
        self.e_fc1_methy_21 = self.fc_layer(input_dim_methy_array[20], level_2_dim_methy)
        self.e_fc1_methy_22 = self.fc_layer(input_dim_methy_array[21], level_2_dim_methy)
        self.e_fc1_methy_X = self.fc_layer(input_dim_methy_array[22], level_2_dim_methy)

        # Level 2
        self.e_fc2_methy = self.fc_layer(level_2_dim_methy*23, level_3_dim_methy)
        # self.e_fc2_methy = self.fc_layer(level_2_dim_methy * 23, level_3_dim_methy, dropout=True)

        # Level 3
        self.e_fc3 = self.fc_layer(level_3_dim_methy, level_4_dim)
        # self.e_fc3 = self.fc_layer(level_3_dim_methy, level_4_dim, dropout=True)

        # Level 4
        self.e_fc4_mean = self.fc_layer(level_4_dim, latent_space_dim, activation=0)
        self.e_fc4_log_var = self.fc_layer(level_4_dim, latent_space_dim, activation=0)

        # DECODER fc layers
        # Level 4
        self.d_fc4 = self.fc_layer(latent_space_dim, level_4_dim)

        # Level 3
        self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_methy)
        # self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_methy, dropout=True)

        # Level 2
        self.d_fc2_methy = self.fc_layer(level_3_dim_methy, level_2_dim_methy*23)
        # self.d_fc2_methy = self.fc_layer(level_3_dim_methy, level_2_dim_methy*23, dropout=True)

        # level 1
        # Methy output for each chromosome
        self.d_fc1_methy_1 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[0], activation=2)
        self.d_fc1_methy_2 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[1], activation=2)
        self.d_fc1_methy_3 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[2], activation=2)
        self.d_fc1_methy_4 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[3], activation=2)
        self.d_fc1_methy_5 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[4], activation=2)
        self.d_fc1_methy_6 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[5], activation=2)
        self.d_fc1_methy_7 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[6], activation=2)
        self.d_fc1_methy_8 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[7], activation=2)
        self.d_fc1_methy_9 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[8], activation=2)
        self.d_fc1_methy_10 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[9], activation=2)
        self.d_fc1_methy_11 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[10], activation=2)
        self.d_fc1_methy_12 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[11], activation=2)
        self.d_fc1_methy_13 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[12], activation=2)
        self.d_fc1_methy_14 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[13], activation=2)
        self.d_fc1_methy_15 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[14], activation=2)
        self.d_fc1_methy_16 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[15], activation=2)
        self.d_fc1_methy_17 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[16], activation=2)
        self.d_fc1_methy_18 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[17], activation=2)
        self.d_fc1_methy_19 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[18], activation=2)
        self.d_fc1_methy_20 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[19], activation=2)
        self.d_fc1_methy_21 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[20], activation=2)
        self.d_fc1_methy_22 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[21], activation=2)
        self.d_fc1_methy_X = self.fc_layer(level_2_dim_methy, input_dim_methy_array[22], activation=2)

        # CLASSIFIER fc layers
        self.c_fc1 = self.fc_layer(latent_space_dim, classifier_1_dim)
        self.c_fc2 = self.fc_layer(classifier_1_dim, classifier_2_dim)
        # self.c_fc2 = self.fc_layer(classifier_1_dim, classifier_2_dim, dropout=True)
        self.c_fc3 = self.fc_layer(classifier_2_dim, classifier_out_dim, activation=0)

    # Activation - 0: no activation, 1: ReLU, 2: Sigmoid
    def fc_layer(self, in_dim, out_dim, activation=1, dropout=False, dropout_p=0.5):
        if activation == 0:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim))
        elif activation == 2:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Sigmoid())
        else:
            if dropout:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
            else:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())
        return layer

    def encode(self, x:list):
        methy_1_level2_layer = self.e_fc1_methy_1(x[0])
        methy_2_level2_layer = self.e_fc1_methy_2(x[1])
        methy_3_level2_layer = self.e_fc1_methy_3(x[2])
        methy_4_level2_layer = self.e_fc1_methy_4(x[3])
        methy_5_level2_layer = self.e_fc1_methy_5(x[4])
        methy_6_level2_layer = self.e_fc1_methy_6(x[5])
        methy_7_level2_layer = self.e_fc1_methy_7(x[6])
        methy_8_level2_layer = self.e_fc1_methy_8(x[7])
        methy_9_level2_layer = self.e_fc1_methy_9(x[8])
        methy_10_level2_layer = self.e_fc1_methy_10(x[9])
        methy_11_level2_layer = self.e_fc1_methy_11(x[10])
        methy_12_level2_layer = self.e_fc1_methy_12(x[11])
        methy_13_level2_layer = self.e_fc1_methy_13(x[12])
        methy_14_level2_layer = self.e_fc1_methy_14(x[13])
        methy_15_level2_layer = self.e_fc1_methy_15(x[14])
        methy_16_level2_layer = self.e_fc1_methy_16(x[15])
        methy_17_level2_layer = self.e_fc1_methy_17(x[16])
        methy_18_level2_layer = self.e_fc1_methy_18(x[17])
        methy_19_level2_layer = self.e_fc1_methy_19(x[18])
        methy_20_level2_layer = self.e_fc1_methy_20(x[19])
        methy_21_level2_layer = self.e_fc1_methy_21(x[20])
        methy_22_level2_layer = self.e_fc1_methy_22(x[21])
        methy_X_level2_layer = self.e_fc1_methy_X(x[22])

        # concat methy tensor together
        methy_level2_layer = torch.cat((methy_1_level2_layer, methy_2_level2_layer, methy_3_level2_layer,
                                        methy_4_level2_layer, methy_5_level2_layer, methy_6_level2_layer,
                                        methy_7_level2_layer, methy_8_level2_layer, methy_9_level2_layer,
                                        methy_10_level2_layer, methy_11_level2_layer, methy_12_level2_layer,
                                        methy_13_level2_layer, methy_14_level2_layer, methy_15_level2_layer,
                                        methy_16_level2_layer, methy_17_level2_layer, methy_18_level2_layer,
                                        methy_19_level2_layer, methy_20_level2_layer, methy_21_level2_layer,
                                        methy_22_level2_layer, methy_X_level2_layer), 1)

        level_3_layer = self.e_fc2_methy(methy_level2_layer)

        level_4_layer = self.e_fc3(level_3_layer)

        latent_mean = self.e_fc4_mean(level_4_layer)
        latent_log_var = self.e_fc4_log_var(level_4_layer)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decode(self, z):
        level_4_layer = self.d_fc4(z)

        level_3_layer = self.d_fc3(level_4_layer)
        # torch.Tensor.narrow(dimension, start, length) 
        methy_level3_layer = level_3_layer.narrow(1, 0, self.level_3_dim_methy)

        methy_level2_layer = self.d_fc2_methy(methy_level3_layer)
        methy_1_level2_layer = methy_level2_layer.narrow(1, 0, self.level_2_dim_methy)
        methy_2_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy, self.level_2_dim_methy)
        methy_3_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*2, self.level_2_dim_methy)
        methy_4_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*3, self.level_2_dim_methy)
        methy_5_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*4, self.level_2_dim_methy)
        methy_6_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*5, self.level_2_dim_methy)
        methy_7_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*6, self.level_2_dim_methy)
        methy_8_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*7, self.level_2_dim_methy)
        methy_9_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*8, self.level_2_dim_methy)
        methy_10_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*9, self.level_2_dim_methy)
        methy_11_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*10, self.level_2_dim_methy)
        methy_12_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*11, self.level_2_dim_methy)
        methy_13_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*12, self.level_2_dim_methy)
        methy_14_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*13, self.level_2_dim_methy)
        methy_15_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*14, self.level_2_dim_methy)
        methy_16_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*15, self.level_2_dim_methy)
        methy_17_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*16, self.level_2_dim_methy)
        methy_18_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*17, self.level_2_dim_methy)
        methy_19_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*18, self.level_2_dim_methy)
        methy_20_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*19, self.level_2_dim_methy)
        methy_21_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*20, self.level_2_dim_methy)
        methy_22_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*21, self.level_2_dim_methy)
        methy_X_level2_layer = methy_level2_layer.narrow(1, self.level_2_dim_methy*22, self.level_2_dim_methy)

        recon_x1 = self.d_fc1_methy_1(methy_1_level2_layer)
        recon_x2 = self.d_fc1_methy_2(methy_2_level2_layer)
        recon_x3 = self.d_fc1_methy_3(methy_3_level2_layer)
        recon_x4 = self.d_fc1_methy_4(methy_4_level2_layer)
        recon_x5 = self.d_fc1_methy_5(methy_5_level2_layer)
        recon_x6 = self.d_fc1_methy_6(methy_6_level2_layer)
        recon_x7 = self.d_fc1_methy_7(methy_7_level2_layer)
        recon_x8 = self.d_fc1_methy_8(methy_8_level2_layer)
        recon_x9 = self.d_fc1_methy_9(methy_9_level2_layer)
        recon_x10 = self.d_fc1_methy_10(methy_10_level2_layer)
        recon_x11 = self.d_fc1_methy_11(methy_11_level2_layer)
        recon_x12 = self.d_fc1_methy_12(methy_12_level2_layer)
        recon_x13 = self.d_fc1_methy_13(methy_13_level2_layer)
        recon_x14 = self.d_fc1_methy_14(methy_14_level2_layer)
        recon_x15 = self.d_fc1_methy_15(methy_15_level2_layer)
        recon_x16 = self.d_fc1_methy_16(methy_16_level2_layer)
        recon_x17 = self.d_fc1_methy_17(methy_17_level2_layer)
        recon_x18 = self.d_fc1_methy_18(methy_18_level2_layer)
        recon_x19 = self.d_fc1_methy_19(methy_19_level2_layer)
        recon_x20 = self.d_fc1_methy_20(methy_20_level2_layer)
        recon_x21 = self.d_fc1_methy_21(methy_21_level2_layer)
        recon_x22 = self.d_fc1_methy_22(methy_22_level2_layer)
        recon_x23 = self.d_fc1_methy_X(methy_X_level2_layer)

        return [recon_x1, recon_x2, recon_x3, recon_x4, recon_x5, recon_x6, recon_x7, recon_x8, recon_x9,
                recon_x10, recon_x11, recon_x12, recon_x13, recon_x14, recon_x15, recon_x16, recon_x17, recon_x18,
                recon_x19, recon_x20, recon_x21, recon_x22, recon_x23]

    def classifier(self, mean):
        level_1_layer = self.c_fc1(mean)
        level_2_layer = self.c_fc2(level_1_layer)
        output_layer = self.c_fc3(level_2_layer)
        return output_layer

    # Half precision training
    # @autocast()
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        classifier_x = mean
        recon_x = self.decode(z)
        if (self.mode == '2stage'):
            pred_y = self.classifier(classifier_x)
            return mean, log_var, z, recon_x, pred_y
        else:
            return mean, log_var, z, recon_x

def rppa_rna_loss(recon_x, x, mean, log_var, pred_y:Tensor=None, y:Tensor=None, mode:str='2stage'):
    # print(recon_x, x)
    recon_ls = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2 - log_var.exp()))
    if (mode == '2stage'):
        classifier_loss = F.cross_entropy(torch.squeeze(pred_y), y.long(), reduction='sum')
        return recon_ls + kl + classifier_loss
    else:
        return recon_ls + kl

def dna_loss(recon_x, x, mean, log_var, pred_y:Tensor=None, y:Tensor=None, mode:str='2stage'):
    # print(recon_x[0].shape, x[0].shape)
    recon_ls = F.binary_cross_entropy(recon_x[0], x[0], reduction='sum')
    for i in range(1, 23):
        recon_ls += F.binary_cross_entropy(recon_x[i], x[i], reduction='sum')
    recon_ls /= 23
    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    if (mode == '2stage'):
        classifier_ls = F.cross_entropy(torch.squeeze(pred_y), y.long(), reduction='sum')
        return recon_ls + kl + classifier_ls
    else:
        return recon_ls + kl 

def segment_x(x:Tensor, chrom_path:str) -> list:
    # sorted by ascending 1-22, X
    import pandas as pd 
    result = [ ]
    chrom23 = ['chr{}'.format(chr) for chr in range(1, 23)]
    chrom23.append('chrX')
    chrom_index = pd.read_csv(chrom_path)
    for chr in chrom23:
        index = chrom_index[chr].tolist()
        result.append(x[:, index])
    return result


