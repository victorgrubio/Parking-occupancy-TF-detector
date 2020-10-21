# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-11-21 12:52:20
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2019-01-17 10:16:01

import argparse
import os
import time

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('folder', help='Folder to maintain clean')
  parser.add_argument('-int','--interval',type=int,help='Interval of cleaning (s)',default=60)
  args = parser.parse_args()

  current_time = time.time()
  print('Initial size of folder: {}'.format(len(os.listdir(args.folder))))
  while True:
    if (len(os.listdir(args.folder)) != 0 and (time.time()-current_time) > args.interval):
       print('Cleaning folder images ...')
       for file in os.listdir(args.folder):
          os.remove(os.path.abspath(args.folder)+'/'+file)
       current_time = time.time()
       print('Folder cleaned successfully')