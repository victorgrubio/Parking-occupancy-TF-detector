# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 13:03:31 2019

@author: gatv
"""
import pyximport;pyximport.install()
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

from utils import ModelProcess as mp

from viewMetrics import plotConfusionMatrix


def getClassificationMetrics(df_metrics,y_true,y_pred,classes,threshold,model_path,plot_metrics,save):
    
    y_pred_th = mp.processPredictions(y_pred,threshold)
    accuracy = metrics.balanced_accuracy_score(y_true,y_pred_th)
    f1 = metrics.f1_score(y_true, y_pred_th)
    conf_matrix = metrics.confusion_matrix(y_true, y_pred_th)
    avg_precision = metrics.average_precision_score(y_true, y_pred)
    final_metrics = pd.DataFrame({
                'threshold': threshold,'model':model_path,
                'accuracy': accuracy,'f1_score': f1,'avg_precision':avg_precision,
                'true_negatives':conf_matrix[0,0],'false_positives':conf_matrix[0,1],
                'false_negatives':conf_matrix[1,0],'true_positives':conf_matrix[1,1]
                 },index=[0])
                 
#    print('\n'.join(['---------------------',
#                     'Threshold Value {}'.format(threshold),
#                     '----------------------'
#                     'f1_score: {}'.format(f1),
#                    'avg_precision: {}'.format(avg_precision),
#                    'confusion_matrix {}'.format(conf_matrix),
#                    ]))
                 
    df_metrics = df_metrics.append(final_metrics,ignore_index=True)
    
    if args.plot_metrics:
        
        cm_name = 'Threshold_{}_CM'.format(threshold)
        cm_norm_name = 'Threshold_{}_CM_NORM'.format(threshold)
        # Plot non-normalized confusion matrix
        plt.figure(cm_name)
        
        plotConfusionMatrix(conf_matrix, classes=classes,
                              title='Confusion matrix, without normalization')
        if save:
            plt.savefig('{}.png'.format(cm_name))
        # Plot normalized confusion matrix
        plt.figure(cm_norm_name)
        plotConfusionMatrix(conf_matrix, classes=classes, normalize=True,
                              title='Normalized confusion matrix')
        if save:
            plt.savefig('{}.png'.format(cm_norm_name))
  
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process a parking image')
    parser.add_argument('dataset_path',help='dataset_path')
    parser.add_argument('-p','--parking',help='parking id',required=True)
    parser.add_argument('-s','--image_size', nargs="+",type=int, help='set size of image:width height',default=[256,256])
    parser.add_argument('--plot_metrics',help='Enable metrics plotting',action='store_true',default=False)    
    parser.add_argument('--save_metrics',help='Save metrics to csv and images',action='store_true',default=False)    
    args = parser.parse_args()
    
    model_dict = {'main':'models/10_10-18_00_normal/',
                  'uk1_lastdense':'models/20_02_19_uk1_lastdense_retrain',
                  'uk1_deep_train': 'models/20_02_19_uk1_deep_retrain'}
                  
    model_name = 'uk1_deep_train'
    model_path = model_dict[model_name]
    model = mp.loadModel(model_path)
    
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(args.dataset_path,
                                target_size=(args.image_size[0], args.image_size[1]),
                                batch_size=32,
                                shuffle=False,
                                interpolation="bicubic")
    
    threshold_list = list(np.arange(0.1, 1.0, 0.1))
    y_pred = model.predict_generator(generator)
    y_true = generator.classes
    df_metrics = pd.DataFrame()
    classes = list(generator.class_indices.keys())
    for threshold in threshold_list:
        threshold = round(threshold,3)
        df_metrics = getClassificationMetrics(df_metrics,y_true,y_pred,classes,threshold,model_name,args.plot_metrics,args.save_metrics)
    
    if args.save_metrics:
        from datetime import datetime
        date = "{:%d_%m_%y-%H_%M}".format(datetime.now())
        path = 'metrics/{}_{}_{}_metrics.csv'.format(model_name,args.parking,date)
        df_metrics.to_csv(path)
        print('Metrics saved at {}'.format(path))
    print(df_metrics)
    