# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 09:38:10 2019

@author: gatv
"""

import pyximport;pyximport.install()
import argparse
import json
import time

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import numpy as np

from utils import FileProcess  as fp

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json



class distroViewer():
    
    def __init__(self,args):
        self.model = self.loadModel()
        self.coord_file  = fp.getCoordFile(args.parking)
        self.data_folder = args.data_folder
        self.args = args
        
    
    """
    Loads model
    """
    def loadModel(self):  
        model_path = 'models/10_10-18_00_normal'
        model_arch_file = model_path+'/model_architecture.json'
        model_weights_file = model_path+'/model_weights.h5py'
        with open(model_arch_file) as json_data:
            model_name_json = json.load(json_data)
            json_data.close()
            
        model = model_from_json(model_name_json)
        model.load_weights(model_weights_file)
    
        return model
    
    """
    Predict using generator in order to retrieve all the images in folder
    """
    def predictFolder(self,image_w=256,image_h=256):
        
        pred_datagen   = ImageDataGenerator(rescale=1./255)
        pred_generator = pred_datagen.flow_from_directory(
            self.data_folder,
            target_size=(image_w,image_h),
            batch_size=64,
            class_mode='binary',
            shuffle=False)
        y_true = pred_generator.classes
        filenames = pred_generator.filenames
        start_time = time.time()
        y_pred = self.model.predict_generator(pred_generator,steps=len(pred_generator),verbose=1).flatten()
        data = {'label':y_true,'pred':y_pred,'filename':filenames}
        df_preds = pd.DataFrame(data=data)
        print('Prediction finished. Time (s): {}'.format(time.time()-start_time))
        return df_preds

    def show(self,df_preds,mode):
        if mode == 'stats':
            self.showDistro(df_preds)
        elif mode == 'images':
            self.showImages(df_preds)
    
    """
    Show resultant distribution, splitted in colours.
    """
    def showDistro(self,df_preds):
        # List of classes to plot
        
        labels = [0,1]
        low_thresh,up_thresh = 0.1,0.9
        bins = int( (up_thresh - low_thresh) / 0.1 )
        df_preds_filter = self.filterDF(df_preds,low_thresh,up_thresh)
        
        # Iterate through labels
        for label in labels:
            # Subset to the class
            subset = df_preds_filter[df_preds_filter['label'] == label]
            
            # Draw the density plot
            sns.distplot(subset['pred'], hist = True, kde = False,
                         bins=bins, label = str(label))
            
        # Plot formatting
        plt.legend(prop={'size': 16}, title = 'Class')
        plt.title(' Histogram / Density Plot of Parking classes')
        plt.xlabel('pred values')
        plt.ylabel('Quantity')
        plt.show()
        
    """
    Show images, class and prediction
    """
    def showImages(self,df_preds):
        
        mode = 'auto'
        low_thresh,up_thresh = 0.1,0.9
        df_preds_filter = self.filterDF(df_preds,low_thresh,up_thresh)
        
        #Iterates over images and show them along with class and prediction
        for index, row in df_preds_filter.iterrows():
            file = '/'.join([self.data_folder,row['filename']])
            lot_image = img.imread(file)
            plt.imshow(lot_image)
            plt.title( '\n'.join([ row['filename'],
                                  'class: {}'.format(row['label']),
                                  'pred: {}'.format(row['pred'])
                                  ]))
            if mode == 'auto':
                plt.pause(1)
            elif mode == 'manual':    
                plt.waitforbuttonpress()

      
    
    def filterDF(self,df_preds,low_thresh,up_thresh):
        return df_preds.loc[(df_preds['pred'] >= low_thresh) &
                            (df_preds['pred'] <= up_thresh)]
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder',help='root folder of classes')
    parser.add_argument('-p','--parking',type=str,help='Parking letter')
    parser.add_argument('--mode',type=str,help='show mode: images or stats (hist,density ...)',default='stats')   
    args = parser.parse_args()
    distro_viewer = distroViewer(args)
    df_preds = distro_viewer.predictFolder()
    distro_viewer.show(df_preds,args.mode)