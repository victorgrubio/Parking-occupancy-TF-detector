# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-11-21 12:52:20
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2019-02-20 13:39:38

import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--all',action='store_true',help='Delete all log folders')
    parser.add_argument('--select',nargs="+",type=str,help='Select which folders are deleted')
    args = parser.parse_args()
    current_folder = os.path.dirname(os.path.abspath(__file__))
    print('Cleaning {} log folders'.format('all' if args.all == True else args.select))
    folders = []
    log_folders = []
    
    # Clean all folders
    if args.all:
        log_folders = [x[0] for x in os.walk(current_folder)]
        #Discard current folder
        log_folders = log_folders[1:]    
        
    # Clean only specified folders
    elif args.select:
        folders = [x[0] for x in os.walk(current_folder)]
        for selected_folder in args.select:
            for log_folder in folders:
                if selected_folder in log_folder:
                    log_folders.append(log_folder)
   
    for log_folder in log_folders:
        for file in os.listdir(log_folder):
            os.remove(os.path.abspath(log_folder)+'/'+file)
        print('Folder {} cleaned successfully'.format(log_folder))