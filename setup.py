# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-10-02 13:03:55
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-12-12 11:35:50

from distutils.core import setup
import os
from shutil import copyfile,copytree,move,rmtree
import tarfile
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dirname  = os.path.dirname(os.path.abspath(__file__))+'/'
release_folder = dirname+'release/'
models_folder  = 'models/10_10-18_00'
config_folder  = '' 
points_folder = ''

def encryptModels():
    return "models/10_10-18_00_encrypted"

def setupGATV(dirname):
    config_folder = 'config/gatv'
    points_folder = 'points/gatv'
    return  config_folder, points_folder

def setupUK(dirname):
    config_folder = 'config/uk'
    models_folder = 'models/10_10-18_00_encrypted_uk'
    return config_folder,models_folder

def setupIndia(dirname):
    config_folder = 'config/india'
    models_folder = 'models/10_10-18_00_encrypted_india'
    return config_folder,models_folder

def configOptions(config_folder,models_folder):
    location = ''
    package_filename = 'parking'
    #Random flag in order to avoid php files
    php_flag = 'fldskjafnmslmfdksañmfldsflaskmdd'
    no_tar = True
    #Customization arguments:
    if '--gatv' in sys.argv:
        config_folder, points_folder = setupGATV(dirname)
        package_filename += '_gatv'
        if '--php' in sys.argv:        
            php_flag = 'php' 
        location = 'gatv'
        sys.argv.remove("--gatv")
    elif '--uk' in sys.argv:
        config_folder, models_folder = setupUK(dirname)
        package_filename += '_uk'
        location = 'uk'
        sys.argv.remove("--uk")
    elif '--india' in sys.argv:
        config_folder,models_folder = setupIndia(dirname)
        package_filename += '_india'
        location = 'india'
        sys.argv.remove("--india")
    if '--encrypted' in sys.argv:
        models_folder = encryptModels()
        package_filename += '_encrypted'
        sys.argv.remove("--encrypted")
    if '--tar' in sys.argv:
        no_tar = False
        sys.argv.remove("--tar")
    return package_filename, config_folder, points_folder, models_folder, php_flag, no_tar, location


py_files  = [
    dirname+"main.py", dirname+'encryptationSSL.py',
    dirname+'parametrizeParking.py', dirname+'setupLogging.py',
    dirname+'utils.py', dirname+'finalProgram.py',
    dirname+'videoThread.py',dirname+'lotsDrawer.py', 
    dirname+'kafka_connector.py']

rmtree(release_folder)
os.makedirs(release_folder)
#Argument processing
package_filename, config_folder, points_folder, models_folder,php_flag, no_tar, location = \
configOptions(config_folder,models_folder)

package_folders = [release_folder, config_folder, dirname+'docker/',dirname+'web_images/',
                   dirname+'points/'+location+'/',dirname+'images/',dirname+'log/']
#Remove old release
rmtree(release_folder)
#Remove .c files and build directory after setup the .so files
files = os.listdir(dirname)
c_cpython_files = [file for file in files if ".c" in file] #.c and .cpython ... .so files
cpython_files = [file for file in files if ".cpython" in file] #.so files only
#Move compiled files to release folder.Also utils,main
if not os.path.isdir(release_folder):
    os.makedirs(release_folder)
    logger.info('Created release folder: {}'.format(release_folder))

required_folders = [models_folder, config_folder, php_flag, points_folder]
#add models and coordinate files to release 
for root, dirs, files in os.walk(".", topdown=False):
    for folder in dirs:
        full_path = os.path.join(root, folder)
        if ( any(required_folder in full_path for required_folder in required_folders) and
              not os.path.isdir(release_folder+full_path) and
              not "release" in full_path and
              not ".git" in full_path):
            copytree(full_path+'/',release_folder+full_path)
            logger.info('Copied file: {}'.format(release_folder+full_path))
#copy .so and .py files
for file in cpython_files+py_files:
    filename = os.path.basename(file)
    release_filename = release_folder+filename
    #If both are the same file, catch exception.
    #Updates release file. Needed due to error on update process
    try:
        copyfile(file,release_filename)
    except:
        os.remove(release_filename)
        copyfile(file,release_filename)
        pass

#remove .so and .c files in main folder
for file in c_cpython_files:
    os.remove(dirname+file)
    logger.info('{} has been removed'.format(file))

if no_tar != True:
    #Remove empty strings from package folders
    package_folders = [x for x in package_folders if x]
    logger.info('Package Folders: {}'.format(package_folders))
    #Create tar file of release folder
    tar_filename = package_filename+'.tar.xz'
    with tarfile.open(tar_filename, 'w:xz') as tar:
        for folder in package_folders:
            current_folder = folder.split("/")[-2]
            logger.info('CURRENT FOLDER: {}'.format(current_folder))
            if 'points' in folder:
                current_folder = 'points/'+folder.split("/")[-2]
            if 'config' in current_folder:
                current_folder = 'config/'+location
            for file in os.listdir(folder):
                if not any(ext in file for ext in ('.git','.log')):
                    logger.info(' Copied {}'.format(current_folder+"/"+file))
                    if 'points' in current_folder:
                        tar.add(current_folder+"/"+file,'release/'+current_folder+"/"+file)
                    else:
                        tar.add(current_folder+"/"+file,current_folder+"/"+file)
            if current_folder == 'images':
                tar.add(current_folder,current_folder)
