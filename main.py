# coding: utf-8

"""
Interpretation: 
    This file used for traning Omi-PGTCN model. One can choose a mode from '2stage' and 'end2end'. 
    The mode 'end2end' will occupy more CPU or GPU source but '2stage' mode don't so, nonetheless, 
    the '2stage' may need relative large running storage and more running time. You can directly to
    run test.py file using pretrained model if you don't want to retrain this model.
"""

import argparse
from myutils import end2end, twostage


args = argparse.ArgumentParser()
args.add_argument('-DN', '--DNA_IN', type=list, default=[37984, 28002, 20127, 15913, 19614, 29135, 24004, 16449, 8027,
                  19759, 23580, 19788, 9737, 12327, 12434, 17943, 23387, 5005, 21032, 8646, 3222, 6987, 9697], 
                  help='The input dimension of DNA Metylation.')
args.add_argument('-RN', '--RNA_IN', type=int, default=18574, help='The input dimension of RNA Expression.')
args.add_argument('-RP', '--RPPA_IN', type=int, default=217, help='The input dimension of Protein Expression.')
args.add_argument('-TH', '--THRESHOLD', type=float, default=0.8, help='The threshold of Pattern Fusion turn to adjacency matrix.')
args.add_argument('-H', '--HOP', type=int, default=10, help='The number of layers of GTCN model.')
args.add_argument('-M', '--MODE', type=str, default='2stage', help='The mode for training model. 2stage or end2end')
args.add_argument('-D', '--DEVICE', type=str, default='cuda', help='The training device. cpu or cuda')
args.add_argument('-LR', '--LR', type=float, default=1e-3, help='The learning rate of end-two-end model.')
args.add_argument('-LR1', '--LR1', type=float, default=1e-3, help='The learning rate of model1.')
args.add_argument('-LR2', '--LR2', type=float, default=1e-3, help='The learning rate of model2.')
args.add_argument('-LR3', '--LR3', type=float, default=1e-3, help='The learning rate of model3.')
args.add_argument('-LR4', '--LR4', type=float, default=1e-3, help='The learning rate of model4.')
args.add_argument('-EP', '--EPOCHS', type=int, default=100, help='The number of training all samples.')
args.add_argument('-BS', '--BATCHSIZE', type=int, default=3, help='The batch size of loading data.')
args.add_argument('-CH', '--CHR_PATH', type=str, default='data_preprocessnig/chrom_index.csv', help='The chrom index path.')
# args.add_argument('-GN', '--GPU_NUMBER', type=int, default=1, help='The number of GPU in you computer. Only set it when -D = gpu.')

opt = args.parse_args()

# print(opt.DNA_IN)

if __name__ == '__main__':
    if(opt.MODE == '2stage'):
        twostage(opt)
    else:
        end2end(opt)

