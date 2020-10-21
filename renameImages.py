"""
Created on Mon Mar 12 10:19:44 2018

@author: victor
"""

import argparse
import os
from lotsDrawer import getParking
from numpy import where

# 5 cifras 
#CORREGIR EL PROBLEMA DEL ORDEN DE LAS IMAGENES (METER 0s)

class editGATVImages():

    def __init__(self,folderPath):
        self.folderPath = folderPath
        self.renamedCounter = 0
        self.deletedCounter = 0
        
    def renameImages(self):

        counter = 0
        extensions = [".jpg",".png",".jpeg"]
        for filename in sorted(os.listdir(self.folderPath)):
            if any(ext in filename for ext in extensions):
                booleanExt = [ext in filename for ext in extensions]
                extPos = where(booleanExt)[0]
                oldName = "".join([self.folderPath,filename])
                newName = oldName
                if args.number:
                    counterZeros = '{0:05d}'.format(counter)
                    newName = self.folderPath+str(counterZeros)+extensions[int(extPos)]
                if args.prefix and args.prefix not in filename:
                    print(args.prefix)
                    newName = self.folderPath+args.prefix+'_'+filename
                if(oldName != newName):
                    print("IMAGEN ",self.renamedCounter," RENOMBRADA")
                    os.rename(oldName,newName)
                    print("oldName:", oldName)
                    print("newName:", newName)
                    self.renamedCounter += 1
                counter += 1    
        print("TOTAL RENOMBRADAS: "+str(self.renamedCounter))
    
    #NO FUNCIONA YA QUE EL 1000 va antes que el 999
    #TIENE QUE TENER X CIFRAS EL VALOR, RESTO: RELLENAR CON CEROS :D    
    def deleteImages(self):
        self.counter = 1
        for filename in sorted(os.listdir(self.imagesPath)):
            if(self.counter > 3795 and ".jpg" in filename):
                #Delete image
                counter_w_zeros = '{0:05d}'.format(self.counter)
                print("IMAGEN "+str(counter_w_zeros)+" ELIMINADA")
                os.remove(self.imagesPath+filename)
                self.counter +=1
                self.deletedCounter += 1
            else:
                print("CONTADOR : "+str(self.counter))
            self.counter +=1
        self.counter = 1
        print("TOTAL RENOMBRADAS: "+str(self.deletedCounter))
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process a parking image')
    parser.add_argument('-f','--folderPath',help='folderPath',required=True)
    parser.add_argument('-d','--delete',help='delete mode',action='store_true')
    parser.add_argument('-n','--number',help='format image number',action='store_true')
    parser.add_argument('-p','--prefix',type=str,help='add prefix',required=True)
    args = parser.parse_args()
    editor = editGATVImages(args.folderPath)
    editor.renameImages()
    if args.delete:
        editor.deleteImages()
