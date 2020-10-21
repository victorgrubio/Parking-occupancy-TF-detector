# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:27:45 2019

@author: gatv
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime


def augment(images_folder,mode):
    """        
    Data augmentation options for ImageDataGenerator:
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None
    """
    # define data preparation
    datagen = ImageDataGenerator(featurewise_center=False,
                                samplewise_center=False,
                                featurewise_std_normalization=False,
                                samplewise_std_normalization=False,
                                zca_whitening=False,
                                zca_epsilon=1e-06,
                                rotation_range=60,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                brightness_range=[0.5,1.5],
                                shear_range=0.0,
                                zoom_range=0.1,
                                channel_shift_range=0.5,
                                fill_mode='nearest',
                                cval=0.0,
                                horizontal_flip=True,
                                vertical_flip=False,
                                rescale=None,
                                data_format='channels_last')
                                
    date = "{:%d_%m_%y-%H_%M_%S}".format(datetime.now())
    if mode == 'save':
        parent_folder  = os.path.dirname(images_folder)
        dataset_folder = parent_folder+'/augdata_'+date+'/'
        print('Generated images will be stored at {}'.format(dataset_folder))
        print('IMPORTANT NOTE: All images will be saved on the same folder, not splitted by classes (yet)')
        os.makedirs(dataset_folder)
        
    elif mode == 'view':
        dataset_folder = None
        
    augmented_flow = datagen.flow_from_directory(
                                directory  = images_folder,
                                batch_size = 16,
                                save_to_dir = dataset_folder,
                                save_prefix = 'aug', save_format='jpeg')
    
    iters  = 0
    #Establish the number of generated images we want to generate for each image
    aug_images_gen = 3
    for batch in augmented_flow:
        if iters >= aug_images_gen*len(augmented_flow):
            break
        else:
            iters += 1
            continue
    
    """
    TO DO: Mejorar modo view para interactuar con batches
    """
    if mode == 'view':
        plt.close('all')
        for data_batch in augmented_flow:
            fig, axes = plt.subplots(nrows=4, ncols=4,num='aug_data')
            for ax, image in zip(axes.flatten(), data_batch[0]):
                image = image.astype(np.uint8)
                ax.imshow(image)
                ax.axis('off')
            # show the plot     
            plt.axis('off')
            plt.show()
            plt.pause(15)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder',help='Path of image folder to augment')
    parser.add_argument('-m','--mode',help='Set mode: view or save',default='save')
    args = parser.parse_args()
    plt.close('all')
    augment(args.image_folder,args.mode)

