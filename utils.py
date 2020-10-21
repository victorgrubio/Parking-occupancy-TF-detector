"""
Created on Mon Jan 29 16:54:43 2018

@author: victor

Loads and draws the contours from a text file which contains the points
of each parking lot
"""
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
from keras.models import model_from_json
import json
import yaml
from encryptationSSL  import decryptFile

class ImgProcess:
    """[summary]
    """
    def loadContours(points_file,img_height=None,img_width=None):
        """
        Load the contours (array of four points) that delimite the lots
        Args: 
            file containing points' coordinates
            img dimensions if needed to resize (points relative to img dims)
        Return: The array of all contours generate by the file content.
        """
        counter = 0
        num_contours = 0
        points_array   = [[0,0],[0,0],[0,0],[0,0]]
        contours_array = np.zeros((1,4,2),dtype=np.int32)
        lines = [line.rstrip() for line in open(points_file) if line.strip() != '']
        #Read each line (point) of the text file
        #Get groups of 4 points (a parking lot) and store it an array  
        for line in lines:
            point = [coord.replace("\n","") for coord in line.split(',')]
            if (float(point[0]) < 1 and float(point[1]) < 1):
                points_array[counter] = [int(float(point[0])*img_width),int(float(point[1])*img_height)]
            else:
                points_array[counter] = [int(float(point[0])),int(float(point[1]))]                
            counter += 1
            #If we have filled the array of points
            #Add a new contours
            #Reset the points array and its counter
            if(counter >= 4):
                contours_array.resize((num_contours+1,4,2), refcheck=False)
                contours_array[num_contours] = np.array(points_array,dtype=np.float32)
                num_contours += 1
                counter = 0 
                points_array= [[0,0],[0,0],[0,0],[0,0]]
        return contours_array
    
    
    def getAngle(p0,p1):
        """
        Get angle between two points
        """
        p0 = list(p0)
        p1 = list(p1)
        alpha = np.arctan([(p1[1]-p0[1])/(p1[0]-p0[0])])
        alpha_deg = alpha*180/(np.pi)
        return float(alpha_deg)
    
    def rotatePoint(point,M): 
        """
        Get rotated points using rotation matrix and original point
        """
        point = np.expand_dims(np.append(point,[1]),axis=1)
        point_rot = np.dot(M,point).astype(int)
        return point_rot
        
    def getLotPoints(img,contour): 
        """
        Get limit points of lot. If black on contour, return None
        """
        max_w,max_h = img.shape[1],img.shape[0]
        rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rect).astype(int)
        for point in box_points:
            if (point[0] > max_w or point[0] < 0
                or point[1] > max_h or point[1] < 0):
                return []
        return box_points
    
    def rotateImage(image, angle):    
        """
        Image rotation without cuts
        """        
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
     
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        if np.abs(angle) >= 45.0:
            angle = 90.0-np.abs(angle)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
     
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
     
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH)), M
    
    def getLotPerspective(contour,img,show=False):     
        """
        Old segmentation method: Perspective based
        """
        x,y,w,h = cv2.boundingRect(contour)
        bound_rect = np.array([[x,y+h],[x,y],[x+w,y],[x+w,y+h]], np.float32)
        M_inter = cv2.getPerspectiveTransform(contour.astype(np.float32),bound_rect)
        mid_img = cv2.warpPerspective(img,M_inter,(x+w,y+h))
        dst = mid_img[y:(y+h-1),x:(x+h-1),:]
        
        img_copy = img.copy()
        contour_list = [contour]
        
        for index,cnt in enumerate(contour_list):
                
            rect = cv2.minAreaRect(cnt)
            # convert all coordinates floating point values to int            
            box = cv2.boxPoints(rect).astype(int)
            # draw a rectangle
            cv2.drawContours(img_copy, [box], 0, (int(255/(index+1)), int(255/(index+1)), 0), 3)
            
            # draw the points of the rectangle                
            for point in cnt:
                cv2.circle(img_copy, (point[0], point[1]), 3, (0, int(255/(index+1)), int(255/(index+1))), -1 )
        if show:
            fig, axes = plt.subplots(3, 1)
            axes[0].imshow(img_copy)
            axes[1].imshow(mid_img)
            axes[2].imshow(dst)
            # remove the x and y ticks
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.show()                
            plt.waitforbuttonpress()
            plt.close()
            
        return dst
    
    def getLotRotate(box_points,img,show=False):    
        """
        New segmentation method: rotations
        """    
        angle = ImgProcess.getAngle(box_points[0],box_points[3])
            
        img_rot,M = ImgProcess.rotateImage(img,angle)
        rot_points = []
        
        for point in box_points:
            rot_point = ImgProcess.rotatePoint(point,M)
            rot_points.append([np.asscalar(rot_point[0]), np.asscalar(rot_point[1])])

        rot_points = np.asarray(rot_points,dtype=int)
        max_point = rot_points.max(0)
        min_point = rot_points.min(0)
        lot_rot = img_rot[min_point[1]:max_point[1],min_point[0]:max_point[0],:]
        
        if show:
            images_list = [img,img_rot,lot_rot]
            for index,img_plot in enumerate(images_list):
                plt.subplot(len(images_list),1,index+1)
                plt.imshow(img_plot)
                plt.axis('off')
            plt.waitforbuttonpress()
            plt.close() 
        return lot_rot
        
    def contour2Image(contour,img,show=False):    
        """
        Transform each array of four coordinates (contour) 
        into an img from a given complete picture.
        Args: Contour and complete img. Show flag for intermediate plots
        Return: img delimited by contour
        """        
        # get the min area rect
        img_copy = img.copy()
        
        box_points = ImgProcess.getLotPoints(img,contour)
        if box_points == []:
            dst = ImgProcess.getLotPerspective(contour,img,show)
        else:
            dst = ImgProcess.getLotRotate(box_points,img,show)
        return dst    
    
    def img2array(img,width=256,height=256,logger=None):    
        """
        Pre-process each parking lot in order to avoid errors in prediction process
        Args: img and final img dimensions (input layer size of model)
        Return: img as array
        """
        if logger != None:
            logger.debug('img processing before model prediction')
        img = cv2.resize(img,(width,height), interpolation = cv2.INTER_CUBIC)
        array = np.asarray(img, dtype='int32')
        array = array / 255
        return array
        
    def drawStatusCircle(img,lot_contour,color):     
        """
        Draws the circle which indicates the occupation of each parking lot
        """
        radio_factor = img.shape[1] / 200
        circle_radio = int((img.shape[1]*radio_factor) / img.shape[0]) #relative to img size
        dict_colors = { 'red':(0,0,255),
                    'green':(0,255,0),
                    'yellow':(0,255,255),
                    'orange':(0,140,255)}
        M = cv2.moments(lot_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        cv2.circle(img, (cX, cY), circle_radio, dict_colors[color], -1)
        return img

    def writeDate(img,date):
        """
        Writes date of prediction on image
        """
        date_img = "{:%H:%M:%S - %d/%m/%y}".format(date)
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = (img.shape[1] /img.shape[0] )*(img.shape[0]/1000)
        thickness = 2
        textSize = cv2.getTextSize(date_img, font, size, thickness)[0]
        cv2.putText(img,date_img,(textSize[1]+2,textSize[1]+2),font,size,(255,200,0),thickness,cv2.LINE_AA)
        return img
        
    def writeParkingStatus(image,predicts):
        """
        Writes predictions of each class in image
        Args: 
            image: Image to draw prediction status
            predicts: List of {slotID,status} dicts
        """
        class_count = {'empty':0,'occupied':0}
        # Loop over predictions, add to each status
        for predict in predicts:
            if predict['status'] == 0:
                class_count['empty'] += 1
            elif predict['status'] == 1:
                class_count['occupied'] += 1
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = (image.shape[1] /image.shape[0] )*(image.shape[0]/750)
        thickness = 1
        vspace = 10
        counter = 1 #To get class index
        # Write on image using loop
        for class_name,preds in class_count.items():
            write_str = '{}: {}'.format(class_name,preds) #Prediction 
            textSize = cv2.getTextSize(write_str, font, size, thickness)[0]
            position = (vspace,image.shape[0]-(textSize[1]*counter + vspace*(counter)) )
            cv2.putText(image,write_str,position,font,size,(255,200,0),thickness,cv2.LINE_AA)
            counter += 1       
        return image
    
    
    def writeLotNumber(image,contour,index):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        write_str = str(index+1) #Prediction 
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = (image.shape[1] /image.shape[0] )*(image.shape[0]/1300)
        thickness = 1
        space = 5
        position = (cX+space,cY)
        cv2.putText(image,write_str,position,font,size,(255,200,0),thickness,cv2.LINE_AA)
        return image
    

class FileProcess:
    """[summary]
    """
    def getCoordFile(zone,parking_id,folder=''):
        return os.path.abspath('{}points/{}/zones_{}.csv'.format(folder,zone,parking_id))
      
    def getConfig(zone,parking_id):  
        """
        Load yaml file and return config dictionary of the specified parking
        """  
        config = {}
        config_file = os.path.abspath('config/{}/{}.yaml'.format(zone,parking_id))
        if os.path.exists(config_file):
            with open(config_file, 'rt') as f:
                config = yaml.safe_load(f.read())
        return config

class ModelProcess:     
    """
    Class containing method realated to models and processing related to predictions
    """    
    
    def loadModel(model_path):
        """
        Load Model, not encrypted
        Args: Path to model (folder containing archicture and weigths)
        Return: Charged Keras Model
        """
        model_name_json    = ''
        model_arch_file    = ''
        model_weights_file = ''
        
        for file in os.listdir(model_path):
            if '.h5py' in file:
                model_weights_file = model_path+'/'+file
            elif '.json' in file:
                model_arch_file = model_path+'/'+file
            if model_weights_file != '' and model_arch_file != '':
                break
        with open(model_arch_file) as json_data:
            model_name_json = json.load(json_data)
            json_data.close()
                
        model = model_from_json(model_name_json)
        model.load_weights(model_weights_file)
        return model


    def processPredictions(y_pred,threshold):
        y_pred_th = []
        for pred in y_pred:
            if pred >= threshold:
                y_pred_th.append(1)
            else:
                y_pred_th.append(0)
        y_pred_th = np.array(y_pred_th,dtype=np.int32)
        return y_pred_th
        
