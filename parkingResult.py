# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:09:51 2019

@author: gatv
"""
from utils import ImgProcess as imp
from utils import FileProcess as fp
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='Path of img of parking')
    parser.add_argument('-p', '--parking', help='Parking name', required=True)
    parser.add_argument('-z', '--zone', help='Parking zone', required=True)
    parser.add_argument(
        '--show', help='Show intermediate image during segmentation', action='store_true')
    parser.add_argument(
        '--test', help='Path of test points file', default=None)
    args = parser.parse_args()

    # Close all previous plots
    plt.close('all')
    img = cv2.imread(args.img)
    img = img[..., ::-1]  # BGR2RGB
    img_copy = img.copy()  # Copy img to draw contours (points)

    # If a test coord file is passed, use it
    if args.test != None:
        coord_file = args.test
    else:
        coord_file = fp.getCoordFile(args.zone, args.parking, 'backup_')

    print('Points file used: {}'.format(coord_file))
    contours = imp.loadContours(coord_file, img.shape[0], img.shape[1])
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 1)  # drawcontours

    # Variable initialization and loop to get the final collage
    rows, cols = 256, 256
    final_img = np.array([])
    imgs_array = []
    dim_limit = 6

    for num_contour in range(0, contours.shape[0]):
        lot_img = imp.contour2Image(contours[num_contour], img, show=args.show)
        lot_img_resized = cv2.resize(
            lot_img, (rows, cols), interpolation=cv2.INTER_CUBIC)
        if num_contour == 0:
            final_img = lot_img_resized
        elif num_contour % dim_limit == 0:
            imgs_array.append(final_img)
            final_img = lot_img_resized
        else:
            final_img = np.concatenate((final_img, lot_img_resized), axis=1)
        if num_contour == contours.shape[0] - 1:
            imgs_array.append(final_img)

    # Plot collage using subplots.
    # Plot the original img and the one with contours drawn
    fig, axes = plt.subplots(len(imgs_array), 1)
    for index in range(0, len(imgs_array)):
        axes[index].imshow(imgs_array[index])
    plt.show()
