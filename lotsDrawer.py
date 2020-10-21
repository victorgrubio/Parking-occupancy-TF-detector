"""
Created on Sat Jan 27 10:07:01 2018

@author: victor
"""
import argparse
import os
import cv2
import numpy as np
from datetime import datetime
from utils import ImgProcess  as ip
from utils import FileProcess as fp

class LotsDrawer:
    #Init method
    def __init__(self,zone,parking_id, contour_width, contour_height,image,display=None):     
        self.coord_file, self.backup_file = self.getCoordFile(zone,parking_id)
        self.contour_width  = contour_width
        self.contour_height = contour_height
        self.num_points     = 0
        self.num_contours   = 0
        self.points_array   = [[0,0],[0,0],[0,0],[0,0]]
        self.backup_array   = [[0,0],[0,0],[0,0],[0,0]]
        self.contours_array = np.zeros((1,4,2),np.int32)
        self.img  = image
        self.height = image.shape[0]
        self.width  = image.shape[1]

        if display != None:
            self.display = True
        else:
            self.display = False    
            
    #obtain coordinates' file path
    def getCoordFile(self,zone,parking_id):
        
        coord_file = fp.getCoordFile(zone,parking_id,folder='')
        backup_file = fp.getCoordFile(zone,parking_id,folder='backup_')
        if os.path.isfile(coord_file):
            coord_file  = 'points/{}/zones_{}_new.csv'.format(zone,parking_id)
            backup_file = 'backup_points/{}/zones_{}_new.csv'.format(zone,parking_id)
        #IF A FILE EXITS, do not override it
        if os.path.isfile(coord_file):
            coord_file  = 'points/{}/test_zones_{}.csv'.format(zone,parking_id)
            backup_file = 'backup_points/{}/test_zones_{}_new.csv'.format(zone,parking_id)
        return coord_file,backup_file
            
    # mouse callback function
    def drawPoints(self,event,x,y,flags,param):
        
        if event == cv2.EVENT_LBUTTONUP:            
            self.points_array[self.num_points] = [x/self.width,y/self.height] 
            if(self.num_points < 3):
                self.num_points += 1
                
            elif(self.num_points >= 3):
                self.points2Txt(self.points_array) #store coordinates
                self.contours_array.resize(self.num_contours+1,4,2)
                self.contours_array[self.num_contours] = self.backup_array
                #draw contours
                mask = np.zeros([self.img.shape[0],self.img.shape[1]],dtype = "uint8")
                cv2.drawContours(mask,self.contours_array,self.num_contours,(1,1,1),-1)
                indv_img = ip.contour2Image(self.contours_array[self.num_contours],self.img,self.display)
                self.num_points = 0
                self.num_contours += 1
            
            
    def points2Txt(self,points_array):
        if(self.num_contours == 0):
            print('Points will be stored at file: {}'.format(self.coord_file))
            file_text   = open(self.coord_file, "w")
            file_backup = open(self.backup_file, "w")
            
        else:
            file_text = open(self.coord_file, "a")
            file_backup = open(self.backup_file, "a")
            
        for i in range(0,4):
            file_text.write( str(self.points_array[i][0])+","+str(self.points_array[i][1])+"\n")
            file_backup.write( str(self.backup_array[i][0])+","+str(self.backup_array[i][1])+"\n")
        
        file_text.close()  
        file_backup.close()
 
    def main(self):
        
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image',self.drawPoints)
        
        while(1):
            cv2.imshow('image',self.img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

