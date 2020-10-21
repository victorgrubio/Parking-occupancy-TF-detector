"""
Created on Mon Jan 29 17:25:59 2018

@author: victor

Script which obtains the img predicted
"""
#allows use of pyx files

import sys
import numpy as np
import cv2
import json
import os
import time
import subprocess
import yaml

from collections import OrderedDict

from datetime import datetime
from keras.models import load_model, model_from_json
from keras import backend as K

from videoThread import VideoThread
from encryptationSSL import decryptFile
from kafka_connector import KafkaConnector

from utils import FileProcess  as fp
from utils import ImgProcess as ip
from utils import ModelProcess as mp


class ParkingDetector:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Avoid unnecesary warning during training

    def __init__(self,args,logger,img=None):
        
        self.logger = logger
        self.img = img
        self.args = args
        self.pred_index = 0
        self.added_host = False
        self.prev_hist_time = time.time()
        self.parking_id = args.parking_id
        self.zone = args.zone
        self.config = fp.getConfig(self.zone, self.parking_id)
        self.model = self.loadModel()
        self.stream_thread = self.initVideoCap()
        self.contours = self.initLots()
        self.predictions = self.initPreds()
        self.kafka_predictions = None

        if args.kafka:
            self.init_kafka()
        #We need to save imgs to send them
        if args.send_results:
            self.args.save_img = True

        #Daily img counter initialized with one of this arguments
        if args.send_results or args.store_pred or args.save_img:
            self.day_imgs = 1
        
        
    def initVideoCap(self):    
        """
        Initialize Video Capture stream
        """
        video_path = self.config['video']['video_path']
        fps_rate = float(self.config['video']['fps_rate'])
        queue_size = int(self.config['video']['queue_size'])
        stream_thread = VideoThread(video_path, self.logger, fps_rate, queue_size)
        #start thread and let it catch images with time
        stream_thread.start()
        #read video capture
        self.logger.info('Starting Video Loop')
        return stream_thread


    def initLots(self):
        """
        Initialize Lots contours based on video streaming dimensions
        """
        tmp_img = self.stream_thread.read()
        while type(tmp_img) != np.ndarray:
            tmp_img = self.stream_thread.read()
        #Init some attributes which need the image dimensions
        #Points, contours and predictions
        points   = self.config['files']['points']
        contours = ip.loadContours(os.path.abspath(points),tmp_img.shape[0],tmp_img.shape[1])
        return contours

    def initPreds(self):
        """
        Init an array to store last X images predictions
        """
        mean_images = int(self.config['video']['mean_images'])
        preds = np.zeros((mean_images,len(self.contours)),dtype=np.float32)
        self.logger.debug('Mean images of prediction (time filtering): {}'.format(mean_images))
        return preds

    def init_kafka(self):
        self.kafka_predictions = KafkaConnector(
                                topic_producer = f"parking.{self.zone}.predictions",
                                group_id=f"parking{self.zone}predictions",
                                bootstrap_servers=[f"{os.getenv('KAFKA_SERVER')}:{os.getenv('KAFKA_PORT')}" ]
                            )
        self.kafka_predictions.init_kafka_producer()

    def getModelPath(self):
        """
        TO DO - MODIFY FOR LOADING FROM CONFIG
        Get the model depending on the parking letter
        """
        if self.args.encrypted == True:
            return os.path.abspath('models/10_10-18_00_encrypted_{}'.format(self.zone))
        else:
            return os.path.abspath('models/10_10-18_00')

 
    def getClassCount(self,prediction):   
        """
        Get number of cases for each prediction class
        """
        class_count = {'empty':0,'occupied':0}
        # Loop over predictions, add to each status
        for predict in prediction:
            if predict['status'] == 0:
                class_count['empty'] += 1
            elif predict['status'] == 1:
                class_count['occupied'] += 1
        self.logger.debug('occupied: {}, empty: {}'.format(
            class_count['occupied'],class_count['empty']))
        return class_count

    def getSavePath(self,prediction=None):
        """
        Get save path of image
        """
        img_abspath = ''
        date_path = 'current/Cam0{}_{:%Y%m%d_%H%M%S}'.format(self.config['video']['cam'],datetime.now())
        #If we need to write results into img filename
        if prediction != None:
            class_count = self.getClassCount(prediction)
            img_filename = '{}_{:02d}_{:02d}.jpg'.format(date_path,class_count['occupied'],class_count['empty'])
        #If we do not (store for training)
        else:
            img_filename = '{}.jpg'.format(date_path)

        img_abspath = self.config['files']['img_dirpath'] + img_filename
        self.logger.debug(img_abspath)
        return img_abspath

    
    def loadModel(self):  
        """
        Loads model
        TO DO - MODIFY TO LOAD FROM CONFIG 
        """
        model_path = self.getModelPath()
        self.logger.debug('Model file: {}'.format(model_path))
        model_name_json = ''
        model_arch_file = model_path+'/model_architecture.json'
        model_weights_file = model_path+'/model_weights.h5py'
        if self.args.encrypted == True:
            print('Password of ARCHITECTURE file')
            decryptFile(model_path+'/model_architecture_encrypted.json',
                        model_arch_file)
            self.logger.info('ARCHITECTURE decrypted success')
            print('Password of WEIGHTS file')
            decryptFile(model_path+'/model_weights_encrypted.h5py',
                        model_weights_file)
            self.logger.info('WEIGHTS decrypted success')
        with open(model_arch_file) as json_data:
            model_name_json = json.load(json_data)
            json_data.close() 
        model = model_from_json(model_name_json)
        model.load_weights(model_weights_file)
        if self.args.encrypted == True:
            os.remove(model_arch_file)
            os.remove(model_weights_file)
        self.logger.info('Model loaded successfully')
        return model
    
    

    def addHost(self):
        """
        Add new host
        """
        known_hosts_filename = '{}/.ssh/known_hosts'.format(os.path.expanduser("~"))
        os.makedirs(os.path.dirname(known_hosts_filename), exist_ok=True)
        with open(known_hosts_filename, 'a') as known_hosts:
            add_known_host_command = 'ssh-keyscan -H {}'.format(self.config['ssh']['server_domain'])
            ssh_process = subprocess.check_call(add_known_host_command.split(),stdout=known_hosts)
            self.added_host = True
        if self.added_host:
            self.logger.info('New host added successfully')       
  
    def updateImgs(self,img,img_abspath,prediction=None): 
        """
        Check wheter we have to update the historical image or not
        """
        if self.args.send_results:
            self.sendCurrentImg(prediction)
        if (time.time() - self.prev_hist_time) > float(self.config['files']['img_save_ratio']):
            if self.args.web:
                self.saveHistImg(img)
            if self.args.send_results:
                self.sendHistImg(img_abspath)
            self.prev_hist_time = time.time()
            if self.day_imgs > 1440:
                self.logger.info('Day images limit reached')
                self.day_imgs = 1
            else:
                self.day_imgs += 1

    def saveImg(self,img,prediction=None):
        """
        Method to save imgs into folders
        """
        img_abspath = self.getSavePath(prediction)
        self.logger.debug('img {} saved'.format(img_abspath))
        img_res = cv2.resize( img, (640,480))
        cv2.imwrite(img_abspath, img_res, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if self.args.web:
            cv2.imwrite('{}/parking{}.jpg'.format(self.config['files']['web_img_dirpath'],self.config['video']['cam']), img_res, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if self.args.save_img:
            cv2.imwrite(img_abspath, img_res, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            #check if we should update the historical image
            self.updateImgs(img_res,img_abspath,prediction)


    def sendCurrentImg(self,prediction):
        """
        Send current img to server
        """ 
        self.logger.debug('Parking CAM: {}'.format(self.config['video']['cam']))
        #Save img before sending it to server
        img_abspath = self.getSavePath(prediction)
        self.logger.debug('img PATH: {}'.format(img_abspath))
        #send img to server
        actual_path = 'parking{}.jpg'.format(self.config['video']['cam'])
        #add known host if not added previously to automatize ssh
        if self.added_host == False:
            self.addHost()
        #send current img to server
        try:
            img_server_path = '{}actual/{}'.format(self.config['ssh']['ssh_server'],actual_path)
            upload_command = 'sshpass -p {} scp {} {}'.format(self.config['ssh']['ssh_key'],img_abspath,img_server_path)
            self.logger.debug('upload command {}'.format(upload_command))
            process = subprocess.check_call(upload_command.split())
        except Exception as e:
            self.logger.error("Error during send current img process: {}".format(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.logger.error(exc_type, fname, exc_tb.tb_lineno)

    def saveHistImg(self,img):
        """
        Save one img per ratio defined in config to show in Today tab
        """
        try:
            img_hist_path = '{}/cam0{}/{}.jpg'.format(
                self.config['files']['web_img_dirpath'],
                self.config['video']['cam'],self.day_imgs)
            cv2.imwrite(img_hist_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            self.logger.info('Hist img written on: {}'.format(img_hist_path))
        except Exception as e:
            self.logger.error("Error during save process of historic img {} from cam {}".format(str(self.day_imgs),str(self.config['video']['cam'])))
            self.logger.error("Error during send Hist img process: {}".format(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.logger.error(exc_type, fname, exc_tb.tb_lineno)

    def sendHistImg(self,img_abspath):
        """
        Send imgs to server
        """
        #Send imgs to historic folder
        try:
            img_server_path = '{}/cam0{}/{}.jpg'.format(self.config['ssh']['ssh_server'],self.config['video']['cam'],self.day_imgs)
            upload_command = 'sshpass -p {} scp {} {}'.format(self.config['ssh']['ssh_key'],img_abspath,img_server_path)
            process = subprocess.check_call(upload_command.split())
            self.logger.info('Historic img number {} from cam {} sent successfully'.format(self.day_imgs,self.config['video']['cam']))
        except Exception as e:
            self.logger.error("Error during the upload of historic img number {} from cam {}".format(self.day_imgs,self.config['video']['cam']))
            self.logger.error("Error during send Hist img process: {}".format(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.logger.error(exc_type, fname, exc_tb.tb_lineno)
    

    def showVideo(self):
        """
        Show video on cv2 window
        """ 
        try:
            cv2.namedWindow(self.parking_id,cv2.WINDOW_NORMAL)
            img = self.img
            cv2.imshow(self.parking_id,img)
            if cv2.waitKey(1) & 0xFF == 27:
                self.logger.warn('Show video stopped along with main execution, due to ESC key pressed')
                self.logger.warn('Stopping thread of frame obtention ...')
                cv2.destroyAllWindows()
                self.stream_thread.stop()
                sys.exit(1)
        except Exception as e:
            self.logger.error('Exception during show video: {}'.format(e))
            self.logger.error("Error during send Hist img process: {}".format(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.logger.error(exc_type, fname, exc_tb.tb_lineno)
            
    def showImg(self):
        win_name = '{}_{}'.format(self.zone,self.parking_id)
        cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
        img = self.img
        cv2.imshow(win_name,img)
        if cv2.waitKey(0) & 0xFF == 27:
            cv2.destroyAllWindows()
            sys.exit(1)

    def normalize_contours(self):
        contours = self.contours.copy()
        for contour in countours:
            contour[0] = contour[0]/self.img.width
            contour[1] = contour[1]/self.img.height
        return contours
    
    def processVideo(self):
        """
        Method for video processing encapsulation at main script
        """
        try:
            if self.args.kafka:
                kafka_config = KafkaConnector(
                                topic_producer = f"parking.{self.zone}.config",
                                group_id=f"parking_{self.zone}_{self.parking_id}_config",
                                bootstrap_servers=[f"{os.getenv('KAFKA_SERVER')}:{os.getenv('KAFKA_PORT')}" ]
                                )
                kafka_config.init_kafka_producer()
                kafka_config.put_data_into_topic({
                    "parking_id": self.parking_id,
                    "zone": self.zone,
                    "lot_contours": self.contours.tolist()
                })
                del kafka_config
            while True:
                #If there is any frame in queue
                if self.stream_thread.more():
                    img = self.stream_thread.read()
                    #If frame exists
                    if type(img) is np.ndarray:
                        prediction_dict = self.predictImg(img.copy()) #predict img
                        #Sends images to kafka broker
                        if self.args.kafka:
                            kafka_message = OrderedDict({
                                "parking_id": self.parking_id,
                                "zone": self.zone,
                            })
                            kafka_message.update(prediction_dict)
                            self.kafka_predictions.put_data_into_topic(data=kafka_message)
                        #Store prediction and original imgs in folder to future training
                        if self.args.store_pred:
                            self.storePredictions(img)
                        # Save images
                        if self.args.save_img or self.args.send_results or self.args.web:
                            self.saveImg(self.img,results['prediction'])
                        # send xml file to update server info
                        if self.args.xml:
                            import xmlProcessing as xml
                            xml.sendGatv(self.parking_id,results['prediction'], self.config['api']['api_server'])
                        # show img without stopping main program   
                        if self.args.display:
                            self.showVideo()
                        # Print state of thread queue
                        if self.args.debug:
                           self.logger.info('QUEUE SIZE: {} . Is full?: {}'.format(self.stream_thread.queue.qsize(),self.stream_thread.queue.full()))
                    # img is not numpy array (probably None)
                    else:
                        self.logger.warn('Type of img is {} instead of numpy ndarray'.format(type(img)))
                        continue
                else:
                    self.logger.debug('No frame in queue')
        except KeyboardInterrupt: #CTRL+C
            self.stream_thread.stop()
            sys.exit(1)
        except Exception as e:
            self.logger.exception(e)

    def predictImg(self,img):
        """
        Predict each parking lot of the parking image and returns the image
        with the circles indicating occupancy
        """
        date = datetime.now().isoformat()
        self.logger.debug('Predict module')
        lot_img_array = []
        threshold = self.config['video']['threshold'] #threshold value for prediction
        id_list = []
        for index,contour in enumerate(self.contours):
            img_lot = ip.contour2Image(contour,img)
            img_lot_normalized = ip.img2array(img_lot,logger=self.logger)
            #Use debugImage method at this point if the cutted image are not correct
            lot_img_array.append(img_lot_normalized)
        lot_img_nparray = np.array(lot_img_array)
        self.logger.debug('batch of predictions moved to numpy array')
        predictions = self.model.predict(lot_img_nparray)
        output_list = []
        results = self.updatePredictions(img,predictions,threshold)
        return results
    
    def updatePredictions(self,img,predictions,threshold): 
        """
        Update the prediction from the last ones, using average of last images
        """

        #update lasts images prediction with last image predicition
        self.predictions[self.pred_index] = predictions.flatten()
        #compute the average for each lot
        self.logger.debug('Time filtering prediction')
        #return results and draw circles based on the average
        avg_predictions = np.mean(self.predictions,axis=0)
        results = self.drawPredicts(img,avg_predictions,threshold)        
        if self.pred_index == len(self.predictions) - 1:
            self.pred_index = 0
        else:
            self.pred_index += 1
        return results
    
    def drawPredicts(self,img,predicts,threshold):     
        """
        Draws the circle which indicates the occupation of each parking lot
        """    
        self.img= img
        date = datetime.now()
        results = OrderedDict({'prediction': [], 'timestamp': date.isoformat()})
        predict_list = []
        margin = 0.5*threshold
        # Loop to iterate over each result and draw the circle
        for index,value in enumerate(predicts):
            #Structure of each prediction: dict object with slotid and status
            predict = OrderedDict([('slotid',index+1),('status',0)])
            M = cv2.moments(self.contours[index])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if(value > threshold):
                if value < (threshold + margin):
                    self.img = ip.drawStatusCircle(self.img,self.contours[index],'orange')
                else:
                    self.img = ip.drawStatusCircle(self.img,self.contours[index],'red')
                    predict['status'] = 1
            else:
                if value > (threshold - margin):
                    self.img = ip.drawStatusCircle(self.img,self.contours[index],'yellow')
                else:
                    self.img = ip.drawStatusCircle(self.img,self.contours[index],'green')
                predict['status'] = 0
            predict_list.append(predict)
            # Write lot number on each lot   
            ip.writeLotNumber(img,self.contours[index],index)
            self.logger.debug('Circle position: X={},Y={}'.format(cX,cY))
            self.logger.debug('Lot {} : {}'.format(index,value))
        results['prediction'] = predict_list
        # Update image attr with the drawn image
        try:
            nodate = self.config['video']['nodate']
        except KeyError:
            self.img = ip.writeDate(img,date)
        self.img = ip.writeParkingStatus(img,results['prediction'])
        self.logger.debug('All prediction circles have been drawn on image.')
        self.logger.debug('self.img updated')
        return results