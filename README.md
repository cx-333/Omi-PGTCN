# Omi-PGTCN Introduction
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/zhangxiaoyu11/OmiVAE/blob/master/LICENSE)
![Safe](https://img.shields.io/badge/Stay-Safe-red?logo=data:image/svg%2bxml;base64,PHN2ZyBpZD0iTGF5ZXJfMSIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgNTEwIDUxMCIgaGVpZ2h0PSI1MTIiIHZpZXdCb3g9IjAgMCA1MTAgNTEwIiB3aWR0aD0iNTEyIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxnPjxnPjxwYXRoIGQ9Im0xNzQuNjEgMzAwYy0yMC41OCAwLTQwLjU2IDYuOTUtNTYuNjkgMTkuNzJsLTExMC4wOSA4NS43OTd2MTA0LjQ4M2g1My41MjlsNzYuNDcxLTY1aDEyNi44MnYtMTQ1eiIgZmlsbD0iI2ZmZGRjZSIvPjwvZz48cGF0aCBkPSJtNTAyLjE3IDI4NC43MmMwIDguOTUtMy42IDE3Ljg5LTEwLjc4IDI0LjQ2bC0xNDguNTYgMTM1LjgyaC03OC4xOHYtODVoNjguMThsMTE0LjM0LTEwMC4yMWMxMi44Mi0xMS4yMyAzMi4wNi0xMC45MiA0NC41LjczIDcgNi41NSAxMC41IDE1LjM4IDEwLjUgMjQuMnoiIGZpbGw9IiNmZmNjYmQiLz48cGF0aCBkPSJtMzMyLjgzIDM0OS42M3YxMC4zN2gtNjguMTh2LTYwaDE4LjU1YzI3LjQxIDAgNDkuNjMgMjIuMjIgNDkuNjMgNDkuNjN6IiBmaWxsPSIjZmZjY2JkIi8+PHBhdGggZD0ibTM5OS44IDc3LjN2OC4wMWMwIDIwLjY1LTguMDQgNDAuMDctMjIuNjQgNTQuNjdsLTExMi41MSAxMTIuNTF2LTIyNi42NmwzLjE4LTMuMTljMTQuNi0xNC42IDM0LjAyLTIyLjY0IDU0LjY3LTIyLjY0IDQyLjYyIDAgNzcuMyAzNC42OCA3Ny4zIDc3LjN6IiBmaWxsPSIjZDAwMDUwIi8+PHBhdGggZD0ibTI2NC42NSAyNS44M3YyMjYuNjZsLTExMi41MS0xMTIuNTFjLTE0LjYtMTQuNi0yMi42NC0zNC4wMi0yMi42NC01NC42N3YtOC4wMWMwLTQyLjYyIDM0LjY4LTc3LjMgNzcuMy03Ny4zIDIwLjY1IDAgNDAuMDYgOC4wNCA1NC42NiAyMi42NHoiIGZpbGw9IiNmZjRhNGEiLz48cGF0aCBkPSJtMjEyLjgzIDM2MC4xMnYzMGg1MS44MnYtMzB6IiBmaWxsPSIjZmZjY2JkIi8+PHBhdGggZD0ibTI2NC42NSAzNjAuMTJ2MzBoMzYuMTRsMzIuMDQtMzB6IiBmaWxsPSIjZmZiZGE5Ii8+PC9nPjwvc3ZnPg==)
[![GitHub Repo stars](https://img.shields.io/github/stars/cx-333/Omi-PGTCN?style=social)](https://github.com/cx-333/Omi-PGTCN/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/cx-333/Omi-PGTCN?style=social)](https://github.com/cx-333/Omi-PGTCN/network/members)
![DOI](https://zenodo.org/badge/doi/001/updating.svg)

 **Paper source code:** Omi-PGTCN: A Whole Process Optimized Model for Pan-cancer Classification and Biomarker Identification.

 **Xin Chen** (cxregion@163.com)

School of Electrical and Information Engneering, ZhengZhou University. 
## Abstract
<p align='justify'>&emsp;&emsp;Tumor, which usually makes people terrified when they hear it due to its low curable rate and high mortality. The reasons for that mainly come from inadequate understanding of a kind of disease. Different types of omics data from one cancer patient have dissimilar insights to the disease, we can obtain more comprehensive views to the malady and thereby better and faster to heal the patients by integrating analysis of multi-omics data. But then, omics data exist “dimension curse” phenomenon that the dimension of omics data normally reaches millions and inversely the number of samples corresponding to omics data is only less than one-tenth. Moreover, it is a nut that applies what method enough to explore the interrelationship among multiple types of omics data. These problems will result in difficult training for machine learning or deep learning model and poor performance of analysis events. Here we proposed a whole process optimized model called Omi-PGTCN to fetch low-dimensional features from multi-omics data and identify important biomarkers, then integrate and concatenate the latent features, respectively, and utilize their similarity and peculiarity information to classify samples. The Omi-PGTCN model combines basic structure of variational auto-encoder and pattern fusion as well as graph tree convolution network to avoid “dimension curse” and “over-fitting” problems and enhance the classification accuracy as well robust of model. The training procedure mainly includes an unsupervised phase in variational auto-encoder and a supervised phase in graph tree convolution network. Omi-PGTCN gained an average classification accuracy of $97.90\pm1.01\%$ after 10-fold cross-validation among 33 types of tumors from the cancer genome atlas program. In addition, the model also identified some important biomarkers that verified by survival analysis of biomarker. The Omi-PGTCN model learned from three types of omics data outperformed the homogeneous models that using one or multiple types of omics data, which shows the complementary information from different omics data types is essential to exhaustive understanding of each type of cancer. </p>

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