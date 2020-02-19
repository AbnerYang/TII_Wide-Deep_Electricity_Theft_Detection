from tensorflow import keras
import numpy as np 
import pandas as pd 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
import os


class MyMetric(keras.callbacks.Callback):

    def __init__(self, train_ratio, num):
        self.train_ratio = train_ratio
        self.num = num
        self.epoch = 0
        
    
    def precision_at_k(self, r, k):

        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def average_precision(self, r):

        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)

    def mean_average_precision(self, rs):

        return np.mean([self.average_precision(r) for r in rs])

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        preds = self.model.predict(self.validation_data[0:2])[:,0]
        y = self.validation_data[2][:,0]
        auc = roc_auc_score(y, preds)
#         print(preds.shape, y.shape)
        
        temp = pd.DataFrame({'label_0':y, 'label_1':1-y, 'preds_0':preds, 'preds_1':1-preds})
        
        map1 = self.mean_average_precision([list(temp.sort_values(by='preds_0',ascending=0).label_0[:100]), 
                                list(temp.sort_values(by='preds_1',ascending=0).label_1[:100])])
        map2 = self.mean_average_precision([list(temp.sort_values(by='preds_0',ascending=0).label_0[:200]), 
                                list(temp.sort_values(by='preds_1',ascending=0).label_1[:200])])
        
        
        print('AUC:%.4f     MAP@100:%.4f      MAP@200:%.4f  \n'%(auc, map1, map2))

        log = 'Epoch:%2d   AUC:%.4f     MAP@100:%.4f      MAP@200:%.4f  \n'%(self.epoch, auc, map1, map2)
        path = 'log/train_ratio_%.1f_num_%d.txt'%(self.train_ratio, self.num)
        if os.path.exists(path):
            fp=open(path,"a+",encoding="utf-8")
            fp.write(log + '\n')
        else:
            fp=open(path,"w",encoding="utf-8")
            fp.write(log + '\n')



        
        
        
 