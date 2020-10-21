# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-07-23 11:48:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-03 09:57:59
import cv2
import os
import argparse


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='create video from images in specified folder')
	parser.add_argument('image_folder', metavar='image_folder', type=str,help='set root of images')
	parser.add_argument('video_name', metavar='video_name', type=str, help='set name of video')
	parser.add_argument('fps', metavar='fps', type=int, help='set fps of video')
	args = parser.parse_args()
	
	init_image = cv2.imread(args.image_folder+'/'+os.listdir(args.image_folder)[1])
	print('Image dims: {}x{}'.format(init_image.shape[1],init_image.shape[0]))
	
	video_w, video_h = 640,480
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	saved_video = cv2.VideoWriter('{}.avi'.format(args.video_name),fourcc, float("{0:.1f}".format(args.fps)), (video_w,video_h))
	print('Video dims: {}x{}'.format(video_w,video_h))
	for image_path in sorted(os.listdir(args.image_folder)):

		image = cv2.imread('/'.join([args.image_folder,image_path]))
		saved_video.write(image)

	saved_video.release()
