# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-11-15 16:26:17
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-11-15 17:04:03
import subprocess
import argparse
from sys import exit

def encryptFile(input_file,output_file):
	encrypt_command = 'openssl enc -aes-256-cbc -in '+input_file+' -out '+output_file
	encrypt_process = subprocess.check_call(encrypt_command.split())

def decryptFile(input_file,output_file):
	output = open(output_file,'w')
	decrypt_command = 'openssl enc -aes-256-cbc -d -in '+input_file
	decrypt_process = subprocess.check_call(decrypt_command.split(),stdout=output)

def encryptFolder(input_folder,output_folder):
	print('Mode not implemented yet')

def decryptFolder(input_folder,output_folder):
	print('Mode not implemented yet')

if __name__ == '__main__':

	parser =  argparse.ArgumentParser()
	parser.add_argument('-m','--mode',type=str,help='Mode: encrypt or decrypt')
	parser.add_argument('input_file', help='Path to input file')
	parser.add_argument('output_file', help='Path to output file')
	args = parser.parse_args()

	if args.mode == 'encrypt':
		encryptFile(args.input_file,args.output_file)
	elif args.mode == 'decrypt':
		decryptFile(args.input_file,args.output_file)
	else:
		print('Mode not implemented ...')
		exit(1)
	
