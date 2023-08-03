# Introduction
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/zhangxiaoyu11/OmiVAE/blob/master/LICENSE)
![Safe](https://img.shields.io/badge/Stay-Safe-red?logo=data:image/svg%2bxml;base64,PHN2ZyBpZD0iTGF5ZXJfMSIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgNTEwIDUxMCIgaGVpZ2h0PSI1MTIiIHZpZXdCb3g9IjAgMCA1MTAgNTEwIiB3aWR0aD0iNTEyIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxnPjxnPjxwYXRoIGQ9Im0xNzQuNjEgMzAwYy0yMC41OCAwLTQwLjU2IDYuOTUtNTYuNjkgMTkuNzJsLTExMC4wOSA4NS43OTd2MTA0LjQ4M2g1My41MjlsNzYuNDcxLTY1aDEyNi44MnYtMTQ1eiIgZmlsbD0iI2ZmZGRjZSIvPjwvZz48cGF0aCBkPSJtNTAyLjE3IDI4NC43MmMwIDguOTUtMy42IDE3Ljg5LTEwLjc4IDI0LjQ2bC0xNDguNTYgMTM1LjgyaC03OC4xOHYtODVoNjguMThsMTE0LjM0LTEwMC4yMWMxMi44Mi0xMS4yMyAzMi4wNi0xMC45MiA0NC41LjczIDcgNi41NSAxMC41IDE1LjM4IDEwLjUgMjQuMnoiIGZpbGw9IiNmZmNjYmQiLz48cGF0aCBkPSJtMzMyLjgzIDM0OS42M3YxMC4zN2gtNjguMTh2LTYwaDE4LjU1YzI3LjQxIDAgNDkuNjMgMjIuMjIgNDkuNjMgNDkuNjN6IiBmaWxsPSIjZmZjY2JkIi8+PHBhdGggZD0ibTM5OS44IDc3LjN2OC4wMWMwIDIwLjY1LTguMDQgNDAuMDctMjIuNjQgNTQuNjdsLTExMi41MSAxMTIuNTF2LTIyNi42NmwzLjE4LTMuMTljMTQuNi0xNC42IDM0LjAyLTIyLjY0IDU0LjY3LTIyLjY0IDQyLjYyIDAgNzcuMyAzNC42OCA3Ny4zIDc3LjN6IiBmaWxsPSIjZDAwMDUwIi8+PHBhdGggZD0ibTI2NC42NSAyNS44M3YyMjYuNjZsLTExMi41MS0xMTIuNTFjLTE0LjYtMTQuNi0yMi42NC0zNC4wMi0yMi42NC01NC42N3YtOC4wMWMwLTQyLjYyIDM0LjY4LTc3LjMgNzcuMy03Ny4zIDIwLjY1IDAgNDAuMDYgOC4wNCA1NC42NiAyMi42NHoiIGZpbGw9IiNmZjRhNGEiLz48cGF0aCBkPSJtMjEyLjgzIDM2MC4xMnYzMGg1MS44MnYtMzB6IiBmaWxsPSIjZmZjY2JkIi8+PHBhdGggZD0ibTI2NC42NSAzNjAuMTJ2MzBoMzYuMTRsMzIuMDQtMzB6IiBmaWxsPSIjZmZiZGE5Ii8+PC9nPjwvc3ZnPg==)
[![GitHub Repo stars](https://img.shields.io/github/stars/cx-333/Omi-PGTCN?style=social)](https://github.com/cx-333/Omi-PGTCN/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/cx-333/Omi-PGTCN?style=social)](https://github.com/cx-333/Omi-PGTCN/network/members)
![DOI](https://zenodo.org/badge/doi/001/updating.svg)

 **Paper source code:** Combining VAE with GTCN for Pan-cancer Prediction and Biomarker Identification using Multi-omics Data.

 **Xin Chen** (cxregion@163.com)

School of Electrical and Information Engneering, ZhengZhou University. 
## Abstract
<p align='justify'>
  
**Objective:**   To overcome the phenomenon of “dimensionality curse” and class imbalance existing in biomolecule data and take full advantage of it to realize pan-cancer prediction and biomarker identification.
  
**Methods:**  We collected a total of 6133 samples with 33 types of tumor from the cancer genome atlas (TCGA), which had been classified and labeled for training, validation and testing, composed of multi-omics data, including DNA methylation, RNA expression and reverse phase protein array (RPPA). Then, we integrate variational autoencoder (VAE) and graph tree convolution network (GTCN) as an entity capable of projecting high-dimensional features to low latent space, generating samples similar to input data, fulfilling pan-cancer prediction and vital biomarkers associated with specific tumor types. The suggested model’s performance was tested following 10-fold cross validation and then compared to main-flow relevant models.

**Results:** The average accuracy of the proposed model reached 93.90±1.01% after 10-fold cross validation in pan-cancer prediction task. Additional relevant evaluation metrics F1 score, precision and recall were 92.03±0.3%, 90.05±1.02% and 91.08±1.0%, respectively. For the prediction performance of single type of cancer, such as LGG and BRCA datasets, the proposed model achieved 83.46±0.19% and 84.02±0.25% prediction accuracy, separately. Furthermore, the model also identified a few essential biomarkers which were proved by the survival curve analysis.

**Conclusion:**  We developed a pan-cancer prediction and biomarker identification system based on jointing VAE and GTCN using multi-omics data. This approach indicates multi-omics data have close relationship with cancer and is conducive to understanding of the mechanism of cancer formation. The predictive results can be consulted by doctors. </p>

# File details

 * my_dataset: Include a class used for loading dataset and three *.txt files storing the name and label of samples.    
 * data_preprocessing: Include some methods that preprocessing dada describerd in the paper.    
 * data_reduce_dimension：The code of VAE model.    
 * data_fusion：The code of PF model.   
 * classification：The code of GTCN model.    
 * manuscript: A paper manuscript file written by *pdf format.    

# Environment requirements (Reference)
The packages that may essential in the file requirements.txt. 


# Data Download 
 * processed data: please contact me.
 * raw data: [DNA Methylatioin](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz); [RNA Expression](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz); [Proteomics](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA-RPPA-pancan-clean.xena.gz)

# Citation
updating...


# License
This source code is licensed under the [MIT](https://github.com/zhangxiaoyu11/OmiVAE/blob/master/LICENSE) license.
