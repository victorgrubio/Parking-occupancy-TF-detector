"""
Created on Sun Jan 28 13:09:42 2018

@author: victor

Generates the individual parking lot dataset 
from parking images and csv files
"""
import argparse
import cv2
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm as progressBar
from keras.preprocessing.image import img_to_array,load_img

from utils import ImageProcess as ip
from utils import FileProcess  as fp
from utils import ModelProcess as mp

class datasetGenerator():
    
    def __init__(self,args):

        self.labels_file    = None
        self.counter_image  = 0
        self.date           = "{:%H_%M-%d_%m_%y}".format(datetime.now())
        self.mode           = args.mode
        self.parking_id     = args.parking_id
        self.images_folder  = args.images_folder
        self.points_file    = fp.getCoordFile(args.parking_id+'_backup')
        self.contours_array = ip.loadContours(self.points_file)
        if self.mode == 'auto':
            self.labels_file = 'ocupaciones/ocupaciones_'+self.parking_name+'.txt'
        elif self.mode == 'predict':
            self.model = mp.loadModel('models/10_10-18_00_normal/')
    """
    Generate lot images from the complete parking
    """  
    def generate(self):
        #Define image extensions in order to avoid conflicts with other files
        extensions = ['.jpg','.png','.jpeg']
        for filename in progressBar(sorted(os.listdir(self.images_folder))):
            if any(ext in filename for ext in extensions):
                image = img_to_array(load_img(self.images_folder+filename))
                image = np.asarray(image,np.float32)
                self.getImagesForDataset(image)

    """
    Obtain  images for the dataset form the complete parking image
    TO DO
    Modify the script to analyse directly the complete contours
    array without any loop. It will increase speed of the process
    """    
    def getImagesForDataset(self,image):
        
        for lot_num,contour in enumerate(self.contours_array):
            lot_image = ip.contour2Image(contour,image)
            self.saveImage(lot_image,lot_num,self.counter_image)
            self.counter_image += 1
     
    """
    Save each lot image into the specific folder with unique filename
    """
    def saveImage(self,lot_image,lot_num_parking,counter_image,status=None):

        test_folder  = 'testdata'
        train_folder = 'traindata'
        # One of each FRACTION_VALUE goes to test
        fraction_test = 5

        # Path of lot image stored
        lot_image_path = ''

        # If we are in auto mode, we use the status of the image
        # and move it to the appropiate folder
        # Default: empty

        # TO-DO: GET STATUS OF THE PARKING LOT USING LABELS FILE!
        classes = {'0': 'empty', '1':'occupied'}
        if self.mode == 'auto' and status != None:
            occ_path = classes[str(status)]
        
        # Position of lot image in the complete dataset: counter_image*lot_num_parking
        lot_num_dataset = counter_image*lot_num_parking
        # Store images in same directory of images folder
        # First we get the parent folder
        parent_folder  = os.path.dirname(os.path.dirname(self.images_folder))
        dataset_folder = parent_folder+'/'+self.parking_id+'_'+self.date
        
        # Folder creation (train and test)
        if not os.path.exists('{}/{}/'.format(dataset_folder,train_folder)):
            print('Images will be generated at {}'.format(dataset_folder))
            os.makedirs('{}/{}/empty'.format(dataset_folder,train_folder))
            os.makedirs('{}/{}/occupied'.format(dataset_folder,train_folder))
            os.makedirs('{}/{}/empty'.format(dataset_folder,test_folder))
            os.makedirs('{}/{}/occupied'.format(dataset_folder,test_folder))
        
        # First we establish if the image is for test or training
        if lot_num_dataset % fraction_test == 0:
            lot_image_path = dataset_folder+'/{}/'.format(test_folder)
        else:
            lot_image_path = dataset_folder+'/{}/'.format(train_folder)

        filename = '_'.join([self.parking_id,
                            str(self.counter_image),
                            str(lot_num_parking)
                            ])
                            
        # Then, using the mode value, we send it to the correspondant folder
        if self.mode == 'auto':
            lot_image_path += '{}/{}'.format(occ_path,filename)
        elif self.mode == 'manual':
            lot_image_path += filename
        elif self.mode == 'predict':
            predicted_folder = self.getPredictedFolder(lot_image,classes)
            lot_image_path += '{}/{}'.format(predicted_folder,filename)
        
        #Saves lot image in file
        cv2.imwrite(lot_image_path+'.jpg', lot_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    """
    Returns the correspondant folder to the image based on the model's prediction
    TO DO
    Adapt it for batches of images
    """    
    def getPredictedFolder(self,image,classes):
        threshold = 0.5
        if self.model != None:
            image_processed = np.expand_dims(ip.preprocessImage(image),axis=0)
            preds = self.model.predict(image_processed)
            preds_processed = mp.processPredictions(preds,threshold)
            for pred in preds_processed:
                return classes[str(pred)]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process a parking image')
    parser.add_argument('-i','--images_folder',help='parking images folder',required=True)
    parser.add_argument('-p','--parking_id',help='Parking id',required=True)
    parser.add_argument('-m','--mode',type=str,help='Set mode to dataset generation: auto or manual')    
    args = parser.parse_args()

    dataset_generator = datasetGenerator(args)
    dataset_generator.generate()
