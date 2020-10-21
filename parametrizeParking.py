# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-11-05 13:42:27
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-11-05 13:48:54

import argparse
import cv2
from lotsDrawer import lotsDrawer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a parking img')
    parser.add_argument('-i','--img_path',help='img_path',required=True)
    parser.add_argument('-p','--parking_letter',help='parking_letter',required=True)
    parser.add_argument('-z','--zone',help='Set parking zone')
    parser.add_argument('-s','--size_img', nargs="+",type=int, help='set size of img:width height',default=[256,256])
    parser.add_argument('-d','--display',help='display mode',action='store_true')
    args = parser.parse_args()
    img = cv2.imread(args.img_path)
    if args.display:
        lotsDrawer = lotsDrawer(args.zone,args.parking_letter,args.size_img[0],args.size_img[1],img,args.display)
    else:
        lotsDrawer = lotsDrawer(args.zone,args.parking_letter,args.size_img[0],args.size_img[1],img)
	
    lotsDrawer.main()
