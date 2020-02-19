from keras.utils import np_utils  
from keras.layers import Input, merge, concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.convolutional import Conv2D, MaxPooling2D  
from keras.models import Model 
from keras.optimizers import SGD

        
def Wide_CNN(weeks, days, channel, wide_len, lr=0.005, decay=1e-5,momentum=0.9):  
    inputs_deep = Input(shape=(weeks*3, days*3, channel))
    inputs_wide = Input(shape=(wide_len,))

    x_deep = Conv2D(32, (3, 3), strides = (3, 3), padding='same', kernel_initializer='he_normal')(inputs_deep)
    x_deep = MaxPooling2D(pool_size=(3, 3))(x_deep)
    x_deep = Flatten()(x_deep)
    x_deep = Dense(128, activation='relu')(x_deep)
    
    x_wide = Dense(128, activation='relu')(inputs_wide)
        
    x = concatenate([x_wide, x_deep])
    x = Dense(64, activation='relu')(x)

    pred = Dense(1, activation='sigmoid')(x)
    
    
    model = Model(inputs=[inputs_wide, inputs_deep], outputs=pred)
    
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  

    model.compile(optimizer=sgd, loss='binary_crossentropy')
    return model 