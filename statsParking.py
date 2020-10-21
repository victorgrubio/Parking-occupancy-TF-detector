"""
Created on Wed Dec 19 11:08:43 2018

@author: visiona
"""

from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt


def getModel(data):  
    kmeans = MiniBatchKMeans(n_clusters=2,random_state=0,batch_size=32,max_iter=10).fit(data)
    return kmeans
    
  
def saveModel(model):
    filename = 'models/thresh_model.sav'
    if os.path.isfile(filename):
        filename = 'models/thresh_model_new.sav'
    joblib.dump(model, filename)
    print('Model saved at: {}'.format(filename))
  
  
def loadModel(filename):
    return joblib.load(filename)

    
def getCentroids(model):
    return model.cluster_centers_

    
def getDistance(sample,model):
    return model.inertia_
    

def predict(input_array,model):
    return model.predict(input_array)
    
    
def plotHist(input_array):
    plt.hist(input_array)
    
    
