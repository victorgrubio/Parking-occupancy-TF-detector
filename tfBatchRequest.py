"""
Created on Fri Mar 15 13:09:03 2019

@author: visiona
"""
import argparse
import os
import time

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations

import tensorflow as tf
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

from keras.preprocessing import image
import numpy as np

if __name__ == '__main__':
    
    # Argument parser for giving input image_path from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", required=True,help="path of the image")
    args = parser.parse_args()
    
    host = 'localhost'
    port = 9000
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'parkingDetector'
    start = time.time()
    image_data = []
    for file in os.listdir(args.image_folder):
        image_path = args.image_folder+file
        with open(image_path, 'rb') as f:
            data = f.read()
            image_data.append(data)

    
    # put data into TensorProto and copy them into the request object
    
    print('In batch mode')
    img_list = []
    img_array = np.array([])
    
    for file in os.listdir(args.image_folder):
        image_path = args.image_folder+file
        # Preprocessing our input image
        img = image.img_to_array(image.load_img(image_path, target_size=(256, 256))) / 255.
        
        # this line is added because of a bug in tf_serving(1.10.0-dev)
        img = np.expand_dims(img.astype('float32'), axis=0)
        if img_array.size != 0:
            img_array = np.concatenate((img_array, img), axis=0)
        else:
            img_array = img
            
    # create TensorProto object for a request

    tensor_proto_make = tf.make_tensor_proto(img_array, shape=[img_array.shape[0], 256, 256, 3])
    print(tensor_proto_make.dtype)
    request.inputs['images'].CopyFrom(tensor_proto_make)

    result = stub.Predict(request, 60.0)
    prediction = result.outputs['prediction'].float_val
    print(prediction)


    print('time elapased: {}'.format(time.time()-start))
