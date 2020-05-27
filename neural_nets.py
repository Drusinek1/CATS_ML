# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:44:17 2020

@author: Dani
"""
import tensorflow as tf
import numpy as np
import pdb

"""
This function will initialize a unet model
input: 
    img: input image 
    @type: 3D ndarray
    num_classes: number of classes in input
    @type: int
    num_level: number of levels in UNet
    @type: int
    num_layers: number of convolutional layers at each level
    @type: int
    kernal_size: size of kernal
    @type: list(int)
    
"""

class UNet_multi:
    def __init__(self, img, filters, features):
        self.model = self.initalize_unet(img, features, filters)
        self.features = features
        filters = filters
    def initalize_unet(self, img, features, filters):
        IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = img

        #img = np.moveaxis(img,0,2)
 
        in_shp = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        #Build the model
        inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), batch_size = 1)
       
        #Convert image integer values into floating point values
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        
        #Contraction Path
        print("input shape = {}".format(inputs.shape))
        c1 = tf.keras.layers.Conv2D(filters*1, (3,3), activation='relu', kernel_initializer='he_normal', padding='same'
                                    , name = "conv_16_1", data_format='channels_last')(s)
        
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(filters*1, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name="conv_16_2", data_format='channels_last')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2,2),name='pool_1', data_format='channels_last')(c1) 
        
        print("layers1 output size = {}".format(p1.shape))
        
        c2 = tf.keras.layers.Conv2D(filters*2, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv_32_1', data_format='channels_last')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(filters*2, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv_32_2', data_format='channels_last')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2,2), data_format='channels_last')(c2)  
       
        print("layers2 output size = {}".format(p2.shape)) 
       
        c3 = tf.keras.layers.Conv2D(filters*3, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv64_1', data_format='channels_last')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(filters*3, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv64_2', data_format='channels_last')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2,2), data_format='channels_last')(c3)  
        
        print("layers3 output size = {}".format(p3.shape)) 
        
        c4 = tf.keras.layers.Conv2D(filters*4, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv128_1', data_format='channels_last')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(filters*4, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv128_2', data_format='channels_last')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2,2), data_format='channels_last')(c4)  
        
        print("layers4 output size = {}".format(p4.shape))
        
        c5 = tf.keras.layers.Conv2D(filters*5, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv256_1', data_format='channels_last')(p4)
        c5 = tf.keras.layers.Dropout(0.2)(c5)
        
        c5 = tf.keras.layers.Conv2D(filters*5, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv256_2', data_format='channels_last')(c5)  

        print("layers5 output size = {}".format(c5.shape))
        
        a5 = tf.keras.layers.Conv2D(filters*6, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv256_1', data_format='channels_last')(c5)
        a5 = tf.keras.layers.Dropout(0.2)(a5)
        
        a5 = tf.keras.layers.Conv2D(filters*6, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv256_3', data_format='channels_last')(a5)  

        print("layers5 output size = {}".format(c5.shape))
        
        """
        Expansive path
        """
        u6 = tf.keras.layers.Conv2DTranspose(filters*5, (2,2), strides=(2,2), padding='same',
                                    name='conv128_4', data_format='channels_last')(c5)
        print("u6 size: {}".format(u6.shape)) 
        
        u6 = tf.keras.layers.concatenate([u6, c4], name='concat1')
            
        
        
         
        u6 = tf.keras.layers.Conv2DTranspose(filters*4, (2,2), strides=(2,2), padding='same',
                                    name='conv128_3', data_format='channels_last')(c5)
        print("u6 size: {}".format(u6.shape)) 
        
        u6 = tf.keras.layers.concatenate([u6, c4], name='concat1')
            

        c6 = tf.keras.layers.Conv2D(filters*4, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv_128_3', data_format='channels_last')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(filters*4, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv128_4', data_format='channels_last')(c6)
        
        print("layers6 output size = {}".format(c6.shape))
       
    

        u7 = tf.keras.layers.Conv2DTranspose(filters*3, (2,2), strides=(2,2), padding='same',
                                             name='deconv1', data_format='channels_last')(c6)
                      
        print("u7 size: {}".format(u7.shape)) 
   
        u7 = tf.keras.layers.concatenate([u7, c3])
        
        print("u7 size: {}".format(u7.shape)) 
        
        
        c7 = tf.keras.layers.Conv2D(filters*3, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv64_3', data_format='channels_last')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        print("c7 = {}".format(c7.shape))
        c7 = tf.keras.layers.Conv2D(filters*3, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv64_4', data_format='channels_last')(c7)

        print("layers7 output size = {}".format(c7.shape))
              
        
        u8 = tf.keras.layers.Conv2DTranspose(filters*2, (2,2), strides=(2,2), padding='same',
                                             name='decov32', data_format='channels_last')(c7)

        u8 = tf.keras.layers.concatenate([u8, c2], name='concat3')
        
        c8 = tf.keras.layers.Conv2D(filters*2, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv32_3', data_format='channels_last')(u8)
        
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(filters*2, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv32_4', data_format='channels_last')(c8)
        
        print("layers8 output size = {}".format(c8.shape))

        u9 = tf.keras.layers.Conv2DTranspose(filters*1, (2,2), strides=(2,2), padding='same',
                                    name='conv16_3', data_format='channels_last')(c8)  
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = tf.keras.layers.Conv2D(filters*1, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv16_4', data_format='channels_last')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(filters*1, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv16_5', data_format='channels_last')(c9)

        #number of filters for last layer must equal number of categories in target
        outputs = tf.keras.layers.Conv2D(features, (1,1), name='conv1', data_format='channels_last')(c9)
        #Outputs logits
        outputs = tf.keras.activations.softmax(outputs, axis=3)
 
        
    

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
        print(model.summary())
        return model
        
    
    
class UNet_binary:
    def __init__(self, img, filters, features):
        self.model = self.initalize_unet(img, features, filters)
        self.features = features
        filters = filters
    def initalize_unet(self, img, features, filters):
        IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = img

        #img = np.moveaxis(img,0,2)
 
        in_shp = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        #Build the model
        inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), batch_size = 1)
       
        #Convert image integer values into floating point values
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        
        #Contraction Path
        c1 = tf.keras.layers.Conv2D(filters*1, (3,3), activation='relu', kernel_initializer='he_normal', padding='same'
                                    , name = "conv_16_1", data_format='channels_last')(s)
        
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(filters*1, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name="conv_16_2", data_format='channels_last')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2,2),name='pool_1', data_format='channels_last')(c1) 
        
        
        c2 = tf.keras.layers.Conv2D(filters*2, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv_32_1', data_format='channels_last')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(filters*2, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv_32_2', data_format='channels_last')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2,2), data_format='channels_last')(c2)  
       
        c3 = tf.keras.layers.Conv2D(filters*3, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv64_1', data_format='channels_last')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(filters*3, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv64_2', data_format='channels_last')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2,2), data_format='channels_last')(c3)  
        
        c4 = tf.keras.layers.Conv2D(filters*4, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv128_1', data_format='channels_last')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(filters*4, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv128_2', data_format='channels_last')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2,2), data_format='channels_last')(c4)  
        
        c5 = tf.keras.layers.Conv2D(filters*5, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv256_1', data_format='channels_last')(p4)
        c5 = tf.keras.layers.Dropout(0.2)(c5)
        
        c5 = tf.keras.layers.Conv2D(filters*5, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv256_2', data_format='channels_last')(c5)  

        a5 = tf.keras.layers.Conv2D(filters*6, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv256_1', data_format='channels_last')(c5)
        a5 = tf.keras.layers.Dropout(0.2)(a5)
        
        a5 = tf.keras.layers.Conv2D(filters*6, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv256_3', data_format='channels_last')(a5)  

        """
        Expansive path
        """
        u6 = tf.keras.layers.Conv2DTranspose(filters*5, (2,2), strides=(2,2), padding='same',
                                    name='conv128_4', data_format='channels_last')(c5)
        
        u6 = tf.keras.layers.concatenate([u6, c4], name='concat1')
            

        u6 = tf.keras.layers.Conv2DTranspose(filters*4, (2,2), strides=(2,2), padding='same',
                                    name='conv128_3', data_format='channels_last')(c5)

        u6 = tf.keras.layers.concatenate([u6, c4], name='concat1')
            

        c6 = tf.keras.layers.Conv2D(filters*4, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv_128_3', data_format='channels_last')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(filters*4, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv128_4', data_format='channels_last')(c6)
        

        u7 = tf.keras.layers.Conv2DTranspose(filters*3, (2,2), strides=(2,2), padding='same',
                                             name='deconv1', data_format='channels_last')(c6)
                      
   
        u7 = tf.keras.layers.concatenate([u7, c3])
        
        c7 = tf.keras.layers.Conv2D(filters*3, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv64_3', data_format='channels_last')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)

        c7 = tf.keras.layers.Conv2D(filters*3, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv64_4', data_format='channels_last')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(filters*2, (2,2), strides=(2,2), padding='same',
                                             name='decov32', data_format='channels_last')(c7)

        u8 = tf.keras.layers.concatenate([u8, c2], name='concat3')
        
        c8 = tf.keras.layers.Conv2D(filters*2, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv32_3', data_format='channels_last')(u8)
        
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(filters*2, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv32_4', data_format='channels_last')(c8)
        

        u9 = tf.keras.layers.Conv2DTranspose(filters*1, (2,2), strides=(2,2), padding='same',
                                    name='conv16_3', data_format='channels_last')(c8)  
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = tf.keras.layers.Conv2D(filters*1, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv16_4', data_format='channels_last')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(filters*1, (3,3), activation='relu', kernel_initializer='he_normal', padding='same',
                                    name='conv16_5', data_format='channels_last')(c9)

        #number of filters for last layer must equal number of categories in target
        outputs = tf.keras.layers.Conv2D(features, (1,1), name='conv1', activation='sigmoid', data_format='channels_last')(c9)
        #Outputs logits

 
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
        
        monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=True,
                                                   restore_best_weights=True)
        #print(model.summary())
        return model
    
    
    
 
    




    
    
    
    