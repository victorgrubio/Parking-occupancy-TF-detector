# -*- coding: utf-8 -*-
# @Author: visiona
# @Date:   2019-03-15 09:54:46
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2019-03-15 10:11:13

import pyximport;pyximport.install()
import os
import json
import argparse
import tensorflow as tf

def loadModel(model_path):
        
        model_name_json    = ''
        model_arch_file    = ''
        model_weights_file = ''
        
        for file in os.listdir(model_path):
            if '.h5py' in file:
                model_weights_file = model_path+file
            elif '.json' in file:
                model_arch_file = model_path+file
            if model_weights_file != '' and model_arch_file != '':
                break
            
        with open(model_arch_file) as json_data:
            model_name_json = json.load(json_data)
            json_data.close()
                
        model = tf.keras.models.model_from_json(model_name_json)
        model.load_weights(model_weights_file)

        return model


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('source_path',help='Path of Keras model')
	parser.add_argument('dst_path',help='Path to save TF model')
	args = parser.parse_args()

	tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

	model = loadModel(args.source_path)
	tf.global_variables_initializer()
	# Fetch the Keras session and save the model
	# The signature definition is defined by the input and output tensors
	# And stored with the default serving key
	with tf.keras.backend.get_session() as sess:
	    tf.saved_model.simple_save(
	        sess,
	        args.dst_path,
	        inputs={'images': model.input},
	        outputs={'prediction': t for t in model.outputs})