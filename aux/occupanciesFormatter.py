# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-07-31 18:54:38
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-03 09:57:58
import argparse
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove


def fileFormatter(filename,origFormat,dstFormat):

	print('The original',origFormat,'are changing to',dstFormat)
	if origFormat == 'TAB':
		origFormat = '\t'
	elif origFormat == 'SPACE':
		origFormat = ' '
	#Create temp file
	fh, abs_path = mkstemp()
	with fdopen(fh,'w') as new_file:
		with open(filename) as old_file:
			for line in old_file:
				new_file.write(line.replace(origFormat, dstFormat))
	#Remove original file
	remove(filename)
	#Move new file
	move(abs_path, filename)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Process a parking image')
	parser.add_argument('-f','--filename',help='file',required=True)
	parser.add_argument('-o','--origFormat',help='parking_letter',required=True)
	parser.add_argument('-d','--dstFormat',help='parking_letter',required=True)
	args = parser.parse_args()
	fileFormatter(args.filename,args.origFormat,args.dstFormat)
