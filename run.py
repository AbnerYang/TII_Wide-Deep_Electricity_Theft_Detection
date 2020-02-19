from datetime import datetime
import time
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from function import *
from keras_metric import *
from wide_cnn import *

if __name__ == '__main__':
    # 1. Read data and label
    print('Read data and label')
    data = pd.read_csv('data/after_preprocess_data.csv')
    label = pd.read_csv('data/label.csv')
    
    
    # 2. Split Train dataset and Test dataset with ratio (50%, 60%, 70%, 80%)
    print('Split Train dataset and Test dataset with ratio (50%, 60%, 70%, 80%)')
    for valr in [0.7, 0.6, 0.5]:
        print('Train split ratio:%.2f'%valr)

        X_train_wide, X_test_wide, Y_train, Y_test = train_test_split(data.values, label.flag.values, test_size=1-valr, random_state = 2017)
        X_train_deep = X_train_wide.reshape(X_train_wide.shape[0],1,-1,7).transpose(0,2,3,1)
        X_test_deep = X_test_wide.reshape(X_test_wide.shape[0],1,-1,7).transpose(0,2,3,1)


        weeks, days, channel = X_train_deep.shape[1], X_train_deep.shape[2], 1
        wide_len = X_train_wide.shape[1]

        print(X_train_wide.shape, X_train_deep.shape)
        print(X_test_wide.shape, X_test_deep.shape)
        

        X_train_pre = self_define_cnn_kernel_process(X_train_deep)
        X_test_pre = self_define_cnn_kernel_process(X_test_deep)
        
        # each model run 10 times and get the avg metric result
        for i in range(10):
            print('Round: %d'%i)
            model=Wide_CNN(weeks, days, channel, wide_len)  

            if i == 0:
                print(model.summary())

            model.fit([X_train_wide, X_train_pre], Y_train, batch_size=64, epochs=30, verbose=1,
                      validation_data=([X_test_wide, X_test_pre], Y_test), callbacks = [MyMetric(valr, i)])










