#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:40:47 2018

@author: victor

Obtains the loss and accuracy as an array in order to do future plots
"""
from keras.callbacks import Callback


class LossHistory(Callback):
    
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_acc  = []

    def on_batch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('acc'))
        self.train_loss.append(logs.get('loss'))
