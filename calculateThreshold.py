# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:52:21 2018

@author: gatv
"""
import pyximport;pyximport.install()
import argparse
#import statsParking as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from lotsDrawer import lotsDrawer
#Class imports from pyx files are made one by one
from utils import FileProcess
from utils import ImageProcess
import sys


def preprocessCSV(df_csv):
    low_thresh = .5
    up_thresh = .8
    df_csv = df_csv.drop([0,len(df_csv.columns)-1],axis=1) #delete first and last column (name and threshold)
    preds_array = df_csv.values
    preds_array = preds_array.flatten()
    values_filter = np.logical_and(preds_array[:] > low_thresh, preds_array[:] < up_thresh)
    indexes = np.where(values_filter)[0]
    return indexes
    
def getImage(parking,images_path,index,df_row,mode):
    
    filename = df_row.iloc[0] # cam_day_time.jpg
    value =  df_row.iloc[index+1] #to compensate the name delete in preprocessing
    date = filename.replace('.jpg','').split('_')[-1]
    parking_image=img.imread(images_path+filename)
    lots_drawer = lotsDrawer(256,256,parking_image,parking,display=True)
    lots_drawer.contours_array = ImageProcess.loadContours(FileProcess.getCoordFile(parking))
    lot_image = ImageProcess.contour2Image(index,lots_drawer.contours_array,parking_image,width=256,height=256)
    plt.imshow(lot_image)
    plt.title( '\n'.join([ 'IMAGE: {}'.format(row),
                           'DATE: {}'.format(date),
                           'LOT: {}'.format(index),
                           'VALUE: {}'.format(value)
                                  ]))
    if mode == 'auto':
        plt.pause(1)
    elif mode == 'manual':    
        plt.waitforbuttonpress()
    else:
        print('Mode not specified, please choose auto or manual')
        sys.exit(1)

    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path',help='Path of csv file containing data')
    parser.add_argument('--mode',type=str,help='Set auto or manual mode',default='auto')
    args = parser.parse_args()
    df_csv = pd.read_csv(args.csv_path,header=None)
    thresh_indexes = preprocessCSV(df_csv)
    parking_tuple = tuple()
    if 'Cam01' in args.csv_path:
        parking_tuple= ('C1','Cam01')
    elif 'Cam02' in args.csv_path:
        parking_tuple = ('C2','Cam02')
   
    images_path = os.path.dirname(args.csv_path)+('/images{}/'.format(parking_tuple[1]))
    total_lots = len(df_csv.columns) -2
    for index in list(thresh_indexes):
        row = index // total_lots
        col = index % total_lots
        df_row = df_csv.iloc[row]
        getImage(parking_tuple[0],images_path,col,df_row,args.mode)