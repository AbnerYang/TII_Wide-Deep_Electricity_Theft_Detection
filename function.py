import numpy as np
import pandas as pd


def expand_data(data):      
    d1 = np.zeros([data.shape[0]*3, data.shape[1], data.shape[2]])
    
    for i in range(data.shape[0]):
        if i >= (data.shape[0] - 2):
            d1[i*3:(i*3+data.shape[0]-i),:,:] = data[i:,:,:]         
        else:
            d1[i*3:(i*3)+3,:,:] = data[i:i+3,:,:]
            
    
    d2 = np.zeros([d1.shape[0], d1.shape[1]*3, d1.shape[2]])
                  
    for j in range(d1.shape[1]):
        if j >= (d1.shape[1] - 2):
            d2[:,j*3:(j*3+d1.shape[1]-j),:] = d1[:,j:,:]         
        else:
            d2[:,j*3:(j*3)+3,:] = d1[:,j:(j+3),:]
    return d2
       
def preprocess_kernel(data):
    data1 = np.zeros(data.shape)
    data2 = np.zeros(data.shape)
    
    for i in range(int(data.shape[0]/3)):
        k = data[(i*3):(i*3+3),:,:]
        data1[i*3,:,:] = 2*k[0,:,:] - k[1,:,:] - k[2,:,:] 
        data1[i*3+1,:,:] = 2*k[1,:,:] - k[0,:,:] - k[2,:,:]
        data1[i*3+2,:,:] = 2*k[2,:,:] - k[0,:,:] - k[1,:,:]
    
    for i in range(int(data.shape[1]/3)):
        k = data[:,(i*3):(i*3+3),:]
        data1[:,i*3,:] = 2*k[:,0,:] - k[:,1,:] - k[:,2,:] 
        data1[:,i*3+1,:] = 2*k[:,1,:] - k[:,0,:] - k[:,2,:]
        data1[:,i*3+2,:] = 2*k[:,2,:] - k[:,0,:] - k[:,1,:]
    
    return data1 + data2
    
        

def self_define_cnn_kernel_process(data):
    '''
    1. expand data from (x, y, z) to (x*3, y*3, z) (Because Conv2D convolution with stride (3,3) for our preprocess)
    
    2. 3*3 kernel process:
    
        [2*V_1 - V_2 - V3
         2*V_2 - V_1 - V3
         2*V_3 - V_1 - V2]

        +

        [2*Vt_1 - Vt_2 - Vt_3, 2*Vt_2 - Vt_1 - Vt_3, 2*Vt_3 - Vt_1 - Vt_2]
      
    '''
    #input
    data_final = np.zeros([data.shape[0], data.shape[1]*3, data.shape[2]*3, data.shape[3]])
    for i in range(data.shape[0]):
        d1 = data[i,:,:,:]
        d1_expand = expand_data(d1)
        d1_final = preprocess_kernel(d1_expand)
        data_final[i,:,:,:] = d1_final
    print(data_final.shape)
    return data_final