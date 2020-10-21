# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:15:15 2019

@author: gatv
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import itertools


def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
def compareRows(df,row_indexes):
    indexes = [value for value in list(df.columns.values) if value not in ['model','threshold']]
    pd.DataFrame({'no train': df.iloc[row_indexes[0]],
                 'retrain': df.iloc[row_indexes[1]]}, index=indexes[:3]).plot(kind="bar",rot=0).legend(bbox_to_anchor=(0.15,1.12))
#    pd.DataFrame({'no train': df.iloc[row_indexes[0]],
#                 'retrain': df.iloc[row_indexes[1]]}, index=indexes[-4:-2]).plot.bar(rot=0)
    pd.DataFrame({'no train': df.iloc[row_indexes[0]],
                 'retrain': df.iloc[row_indexes[1]]}, index=indexes[-4:]).plot(kind="bar",rot=0).legend(bbox_to_anchor=(0.15, 1.12))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',help='save resultant metrics',action='store_true',default=False)
    args = parser.parse_args()
    plt.close('all')
    
    #TO DO - Read from folder
    csv_dict = {'notrain': 'metrics/main_UK1_22_02_19-12_07_metrics.csv',
                'deep_retrain': 'metrics/uk1_deep_train_UK1_22_02_19-13_57_metrics.csv'
    }
    df_notrain = pd.DataFrame.from_csv(csv_dict['notrain'])
    df_retrain = pd.DataFrame.from_csv(csv_dict['deep_retrain'])
    # TO DO - Do it iteratively. Store previous on list
    df_main = df_notrain.append(df_retrain).reset_index()
    df_main = df_main.drop(['index'],axis=1)
    del df_notrain,df_retrain
    column_names = list(df_main.columns.values)
    models = list(df_main.model.unique())
    best_rows_indexes = []
    for model in models:
        df_model = df_main[(df_main['model'] == model)]
        best_rows_indexes.append(df_model[df_model['accuracy']==df_model['accuracy'].max()].index[0])
    
    compareRows(df_main,best_rows_indexes)
        
    print('Finished')
