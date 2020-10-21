# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-10-03 10:52:34
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-12-11 12:25:43

import sys
import os
import cv2
import argparse
from finalProgram import ParkingDetector
import logging

from setupLogging import setupLogging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Avoid unnecesary warning during training

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process a parking img')
    parser.add_argument('-p','--parking_id',help='parking_letter',required=True)
    parser.add_argument('-i','--img_path',help='img_detection')
    parser.add_argument('-z','--zone', help='zone of parking', required=True)
    parser.add_argument('--save_img',help='save imgs of execution',action='store_true',default=False)
    parser.add_argument('--web',help='send imgs to web',action='store_true',default=False)    
    parser.add_argument('--send_results',help='send imgs to remote server',action='store_true',default=False)
    parser.add_argument('--xml',help='generate xml during predictions',action='store_true',default=False)
    parser.add_argument('--display',help='show img on cv2 window',action='store_true',default=False)
    parser.add_argument('--debug',help='debug mode',action='store_true',default=False)
    parser.add_argument('--test',help='test algorithm precision',action='store_true',default=False)
    parser.add_argument('-st','--store_pred',help='store imgs and predictions for training',action='store_true',default=False)
    parser.add_argument('-enc','--encrypted',help='mode for encrypted model',action='store_true',default=False)
    parser.add_argument('--kafka', help='add kafka communication',action='store_true',default=False)
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)    
    setupLogging()
    
    try:
        #Process Video
        if args.img_path != True:
            detector = ParkingDetector(args,logger)
            detector.processVideo()

        #Process one img
        else:
            img = cv2.imread(args.img_path)
            detector = parkingDetector(args,logger,img)
            detector.stream_thread.stop()
            detector.img = img
            results = detector.predictImg()
            detector.showImg()
            if args.xml:
                detector.sendXML(results)
            if args.display:
                detector.showimg()
    except KeyboardInterrupt as e: #CTRL+C
        logger.error('Exit program using CTRL+C. Details: {}'.format(e))
        sys.exit(1)
