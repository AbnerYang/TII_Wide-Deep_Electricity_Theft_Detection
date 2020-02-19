# TII_Wide&CNN_Electricity_Theft_Detection
This is the source code of our paper on electricity-theft detection published in TII in the 2017 year.


- Zibin Zheng , Yatao Yang , Xiangdong Niu , Hong-Ning Dai, Yuren Zhou. Wide & Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids[J]. IEEE Transactions on Industrial Informatics, 2017:1-1.


# Source Code Introduction

 - function.py : include custom data processing functions that are needed in the experiment.
 
 - keras_metric.py : include the AUC, MAP@100, and MAP@200 metric function which will be executed on each epoch.
 
 - wide_cnn.py: the source code of our Wide&CNN model.
 
 - run.py : the main file, we can run to get the experimental results.
 
 - log/ : store experimental result logs.
 
 - data/ : store experimental datasets.
 
 # Dataset Download Address
 
 LINK: https://pan.baidu.com/s/17exE465yp79HWJ06qRvWKg Extraction code: i6xy
 
 It should be noted that tha data contains two files. 
 
  - after_preprocess_data.csv : it is the electricity consumption data of users after data preprocessing. 
  
  - label.csv : it is the label  data of whether a user steal electricity. Each line corresponds to each user.
 
 
 
 
 
 
