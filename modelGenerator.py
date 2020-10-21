# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:25:59 2018

@author: victor
"""


import pyximport;pyximport.install()
import argparse
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import time

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential, model_from_json
from keras.layers import Activation, Dense, Flatten, Input, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from clr_callback import CyclicLR
#obtener las dimensiones, el directorio y el numero de ejemplos, exportando un objeto dataset
        
class modelGenerator():
    
    def __init__(self,args,num_plaza=None,parking_letter=None):        
        self.image_width  = args.size_image[0]
        self.image_height = args.size_image[1]
        self.dataset_path = args.dataset_path
        self.epochs       = args.epochs
        self.batch_size   = args.batch_size
        self.version      = args.version
        self.date         = "{:%d_%m_%y-%H_%M}".format(datetime.now())
        self.loss         = 'binary_crossentropy'
        self.optimizer    = Adam(lr=0.001)
        self.metrics      = 'accuracy'
        self.class_mode   = 'binary'
        self.model_path   = 'models/'+self.date+'/'
    
    """
    Create data generators needed to train
    """    
    def createDataGenerator(self):
        # This is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator( rescale=1./255)
        
        # This is the augmentation configuration we will use for testing:
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path+'traindata',
            target_size=(self.image_width, self.image_height),
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            interpolation="bicubic")
        
        validation_generator = test_datagen.flow_from_directory(
            self.dataset_path+'testdata',
            target_size=(self.image_width, self.image_height),
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            interpolation="bicubic")
        
        return train_generator, validation_generator

    """
    Create new model (architecture)
    """
    def createModel(self):
        
        if K.image_data_format() == 'channels_first':
            input_shape = (3, self.image_width, self.image_height)
        else:
            input_shape = (self.image_width, self.image_height, 3)
        
        if self.version == 'v0':
            return self.loadV0(input_shape)
        elif self.version == 'v1':    
            return self.loadV1(input_shape)
            
    """
    Load first version of the parking's model architecture
    """
    def loadV0(self,input_shape):
        #Creating the model
        model = Sequential()
        
        #CNN1
        model.add(Conv2D(32, (3, 3),input_shape=input_shape,name='conv2d_1'))
        model.add(Activation('relu',name='relu_1'))
        model.add(MaxPooling2D(pool_size=(2, 2),name='pool_1'))
        
        #CNN2
        model.add(Conv2D(32, (3, 3),name='conv2d_2'))
        model.add(Activation('relu',name='relu_2'))
        model.add(MaxPooling2D(pool_size=(2, 2),name='pool_2'))
        
        #CNN3
        model.add(Conv2D(64, (3, 3),name='conv2d_3'))
        model.add(Activation('relu',name='relu_3'))
        model.add(MaxPooling2D(pool_size=(2, 2),name='pool_3'))
        
        #Densely connected layers
        model.add(Flatten(name='flatten_1'))
        model.add(Dense(64,name='dense_1'))
        model.add(Activation('relu',name='relu_4'))
        model.add(Dropout(0.5,name='dropout_1'))
        model.add(Dense(1,name='dense_2'))
        model.add(Activation('sigmoid',name='sigmoid_final'))
    
        print(model.summary())
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=[self.metrics])
        
        return model

    """
    Load second version of the parking's model architecture
    """
    def loadV1(self,input_shape):
        
        #In functional Keras model, the previous layer must be specified
        #after the layer declarations
        image_input = Input(shape=input_shape,name='input')
    
        #CNN1
        current_layer = Conv2D(32, (3, 3),name='conv2d_1',padding='same')(image_input)
        current_layer = Activation('elu',name='activation_1')(current_layer)
        current_layer = BatchNormalization(axis=2,name='bn1')(current_layer)
        current_layer = MaxPooling2D(pool_size=(2, 2),name='pool_1')(current_layer)
        
        #CNN2
        current_layer = Conv2D(64, (3, 3),name='conv2d_2',padding='same')(current_layer)
        current_layer = Activation('elu',name='activation_2')(current_layer)
        current_layer = BatchNormalization(axis=2,name='bn_2')(current_layer)
        current_layer = MaxPooling2D(pool_size=(2, 2),name='pool_2')(current_layer)
        
        #CNN3
        current_layer = Conv2D(128, (3, 3),name='conv2d_3',padding='same')(current_layer)
        current_layer = Activation('elu',name='activation_3')(current_layer)
        current_layer = BatchNormalization(axis=2,name='bn_3')(current_layer)
        current_layer = MaxPooling2D(pool_size=(2, 2),name='pool_3')(current_layer)
        
        #CNN4
        current_layer = Conv2D(256, (3, 3),name='conv2d_4',padding='same')(current_layer)
        current_layer = Activation('elu',name='activation_4')(current_layer)
        current_layer = BatchNormalization(axis=2,name='bn_4')(current_layer)
        current_layer = MaxPooling2D(pool_size=(2, 2),name='pool_4')(current_layer)
        
        #CNN5
        current_layer = Conv2D(512, (3, 3),name='conv2d_5',padding='same')(current_layer)
        current_layer = Activation('elu',name='activation_5')(current_layer)
        current_layer = BatchNormalization(axis=2,name='bn_5')(current_layer)
        current_layer = MaxPooling2D(pool_size=(2, 2),name='pool_5')(current_layer)
        
        #FC1
        current_layer = Flatten(name='flatten_1')(current_layer)
        current_layer = Dense(256,name='dense_1')(current_layer)
        current_layer = Activation('elu',name='activation_6')(current_layer)
        current_layer = BatchNormalization(name='bn_6')(current_layer)

        #FC2 and output
        current_layer = Dense(128,name='dense_2')(current_layer)
        current_layer = Activation('elu',name='activation_7')(current_layer)
        current_layer = BatchNormalization(name='bn_7')(current_layer)
        current_layer = Dense(1,name='dense_3')(current_layer)
        final_output  = Activation('sigmoid',name='sigmoid_final')(current_layer)
        
        model = Model(inputs=image_input, outputs=final_output)
        print(model.summary())
        model.compile(loss=self.loss,optimizer=self.optimizer, metrics=[self.metrics])
        return model
    
    """
    TO DO
    Load Resnet-v2 model from keras with pretrained weights and frozen layers
    """ 
    def loadResnetV2(self,input_shape):
        print('ResNet NOT IMPLEMENTED')
    
    
    """
    Load pre-trained model
    """
    def loadModel(self,model_path):
        
        with open(model_path+'model_architecture.json') as json_data:
            model_name_json = json.load(json_data)
            json_data.close()
        model = model_from_json(model_name_json)
        model.load_weights(model_path+'model_weights.h5py')
        return model
    

    """
    Retrain model using transfer Learning
    """
    def retrainModel(self):
        
        #Freeze the densely connected layers (last five)
        #Print only layers to be trained        
        if self.version == 'v0':
            model_path = "models/10_10-18_00_normal/"
            model      = self.loadModel(model_path)
            for layer in model.layers[:-5]:
                layer.trainable = False
            for layer in model.layers[-5:]:
                print(layer.name)
                layer.trainable = True
        
        #TO DO: change model path   
        elif self.version == 'v1':
            model_path = "models/10_10-18_00_normal/"
            model      = self.loadModel(model_path)
            for layer in model.layers[:-5]:
                layer.trainable = False
            for layer in model.layers[-5:]:
                print(layer.name)
                layer.trainable = True
        
        #TO DO: Define retrain layers for ResnetV2 and change model path
        elif self.version == 'resnet-v2':
            model_path = "models/10_10-18_00_normal/"
            model      = self.loadModel(model_path)
            for layer in model.layers[:-5]:
                layer.trainable = False
            for layer in model.layers[-5:]:
                print(layer.name)
                layer.trainable = True
            
        #compile model
        model.compile(loss=self.loss,
                  optimizer=self.optimizer,
                  metrics=[self.metrics])
        print(model.summary())
        
        return model
    
    """
    Training
    """
    def trainModel(self,model,train_generator,validation_generator):

        os.makedirs(self.model_path)
        model_json = model.to_json()
        with open(self.model_path+'/model_architecture.json', 'w') as outfile:
            json.dump(model_json, outfile)
        
        #Callback definition
        
        checkpoint = ModelCheckpoint(self.model_path+"/model_weights_{epoch:04d}.h5py",
                                     monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='auto')
                                     
        csv_logger = CSVLogger(self.model_path+'/csvLogger_'+self.date+'.csv',
                               separator=',', append=True)
                               
        tboard = TensorBoard(log_dir=self.model_path,
                                 write_grads=True, write_images=True,
                                 batch_size=self.batch_size, write_graph=False,
                                 embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None,
                                 histogram_freq=0)
                                 
        clr = CyclicLR(base_lr=0.00001, max_lr=0.00006,
                        step_size=8*len(train_generator))
        
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=20, verbose=1, mode='auto',
                                   baseline=None, restore_best_weights=True)
        
        callbacks_list = [checkpoint,csv_logger,tboard,clr,early_stop]
        
        # Training/Test phase using fit_generator
        start_time = time.time()
        history = model.fit_generator(train_generator,
                            steps_per_epoch=len(train_generator),
                            epochs=self.epochs,
                            validation_data=validation_generator,
                            validation_steps=len(validation_generator),
                            callbacks=callbacks_list)
        
        training_time = print('Training duration: {}'.format(time.time() - start_time))
        
        return history,training_time
        
    """
    Plot results (probably not needed now that we have tensorboard)
    """    
    def plotResults(self,history,path):
        
        self.saveResults(history,path)
        # summarize history for accuracy
        plt.plot(history.train_acc)
        plt.title('train accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('batch')
        plt.ylim([0.5,1])
        plt.savefig(path+'/modelAccuracy.png',dpi=100)
        plt.show()
        plt.plot(history.train_loss)
        plt.title('train loss')
        plt.ylabel('loss')
        plt.xlabel('batch')
        plt.ylim([0,3])
        plt.savefig(path+'/modelLoss.png',dpi=100)
        plt.show()
        
    def save_model(self,model,history,training_time,transf_learn_flag=None):
        
        if(transf_learn_flag != None):
            transf_learn_flag = 'YES'
        else:
            transf_learn_flag = 'NO'

        model_json = model.to_json()
        with open(self.model_path+'model_architecture.json', 'w') as outfile:
            json.dump(model_json, outfile)    
        train_results_file = self.model_path+"/train_results.txt"
        # Guardamos los datos del modelo entrenado en un archivo txt
        with open(train_results_file, "w") as fileTxt:
            
            fileTxt.write('\n'.join([
                'Dataset: {}'.format(self.dataset_path),
                'Image_size: {}x{}'.format(self.image_width,self.image_height),
                'Batch Size: {}'.format(self.batch_size),
                'Training time: {}'.format(training_time),
                'Epochs: {}'.format(self.epochs),
                'Model loss: {}'.format(self.loss),
                'Model optimizer: {}'.format(self.optimizer),
                'Model metrics: {}'.format(self.metrics),
                'Transfer learning? : '+ transf_learn_flag
                ]))
        
        print('MODELO GUARDADO CORRECTAMENTE')
    
    
    """
    Main method
    """    
    
    def main(self,transf_learn_flag=None):
        
        if(transf_learn_flag != None):
            model = self.retrainModel()
        else:
            model = self.createModel()
            
        train_generator,test_generator = self.createDataGenerator()
        
        history,training_time = self.trainModel(model,train_generator,test_generator)
        self.save_model(model,history,training_time,transf_learn_flag)     
        K.clear_session()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process a parking image')
    parser.add_argument('-d','--dataset_path',help='dataset_path',required=True)
    parser.add_argument('-v','--version',type=str,help='version of model architecture: v0,v1 ...',default='v0')
    parser.add_argument('-s','--size_image', nargs="+",type=int, help='set size of image:width height',default=[256,256])
    parser.add_argument('-e','--epochs',type=int, help='set number of epochs',default=10)
    parser.add_argument('-b','--batch_size',type=int, help='set batch size',default=32)
    parser.add_argument('-t','--transf', help='set transfer learning mode',action='store_true',default=False)
    args = parser.parse_args()

    model = modelGenerator(args)
    
    if args.transf:
        model.main('transf')
    else:
        model.main()
