# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:33:10 2019

@author: gatv
"""
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np

def lotDetection(parking_img):
    canny_th = [100,255]
    canny_img = cv2.Canny(parking_img, canny_th[0], canny_th[1], 3)
    canny_img_bgr = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
    minLineLength = 10
    maxLineGap = 10
    lines = cv2.HoughLinesP(canny_img,1,np.pi/180,7,minLineLength,maxLineGap)
    
    for i in range(lines.shape[0]):
        line = lines[i].flatten()
        cv2.line(canny_img_bgr,(line[0],line[1]),(line[2],line[3]),(0, 255, 0), 3)
    
    plt.figure('canny')
    plt.imshow(canny_img,cmap='gray')
    plt.axis('off')
    plt.figure('hough lines')
    plt.imshow(canny_img_bgr,cmap='gray')
    plt.axis('off')
    plt.show()

def hsvViewer(img,mode='bgr'):
    if mode == 'bgr':
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    elif mode == 'gray':
        hsv_img = cv2.cvtColor(img,cv2.COLOR_GRAY2HSV)
    for i in range(hsv_img.shape[2]):
        plt.figure(i)
        plt.imshow(hsv_img[:,:,i])
        plt.axis('off')
    plt.show()

def preprocessValue(img_value,show=False):
    alpha = 1
    beta = -40.0
    new_img = cv2.convertScaleAbs(img_value, alpha=alpha, beta=beta)
    if show:    
        plt.figure('value')
        plt.imshow(img_value)
        plt.axis('off')
        plt.colorbar()
        plt.figure('value_mod')
        plt.imshow(new_img,cmap='gray')
        plt.axis('off')
        plt.colorbar()
        plt.show()
    return new_img

def adjust_gamma(img, gamma=10.0, show=False):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    table = np.array([((i / 255.0) ** gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    # apply gamma correction using the lookup table
    adjusted = cv2.LUT(img, table)
    if show:
        plt.figure('adjusted')
        plt.imshow(adjusted)
        plt.axis('off')
        plt.figure('adjusted_gray')
        plt.imshow(adjusted,cmap='gray')
        plt.axis('off')
        
        plt.show()
    return adjusted

def nothing(x):
    pass
 
def trackbarCanny(parking_img):
    img = cv2.blur(parking_img, (7,7))
     
    canny_edge = cv2.Canny(img, 0, 0)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('canny_edge', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.imshow('canny_edge', canny_edge)
     
    cv2.createTrackbar('min_value','canny_edge',0,500,nothing)
    cv2.createTrackbar('max_value','canny_edge',0,500,nothing)
     
    while(True):
        cv2.imshow('img', img)
        cv2.imshow('canny_edge', canny_edge)
         
        min_value = cv2.getTrackbarPos('min_value', 'canny_edge')
        max_value = cv2.getTrackbarPos('max_value', 'canny_edge')
     
        canny_edge = cv2.Canny(img, min_value, max_value)
         
        k = cv2.waitKey(37)
        if k == 27:
            break
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    plt.close('all')
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path',help='Path to parking img')
    args = parser.parse_args()
    img = cv2.imread(args.img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_value = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2]
    img_value_proc = preprocessValue(img_value,show=False)
    adjusted = adjust_gamma(img_value_proc, gamma=2.5, show=True)
    #hsvViewer(img)
    trackbarCanny(adjusted)
    #lotDetection(img_gray)
    