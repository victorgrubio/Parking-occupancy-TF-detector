# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2019-03-15 10:38:08
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2019-03-17 19:39:31
import argparse
import json
import os
import numpy as np
import requests
from keras.preprocessing import image

if __name__ == '__main__':
    # Argument parser for giving input image_path from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", required=True,help="path of the image")
    args = parser.parse_args()
    img_list = []
    img = None
    
    for file in os.listdir(args.image_folder):
        image_path = args.image_folder+file
        # Preprocessing our input image
        img = image.img_to_array(image.load_img(image_path, target_size=(256, 256))) / 255.
        print(img.shape)
        # this line is added because of a bug in tf_serving(1.10.0-dev)
        img = img.astype('float16')
        img_list.append(img)
        #print(len(img.tolist()))
        #print(img.tolist())

    img_array = np.array(img_list)
    print(img_array.shape)
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }
    
    # sending post request to TensorFlow Serving server
    r = requests.post('http://localhost:9000/v1/models/parkingDetector:predict', json=payload)
    pred = json.loads(r.content.decode('utf-8'))
    print(pred)
    
    # Decoding the response
    # decode_predictions(preds, top=5) by default gives top 5 results
    # You can pass "top=10" to get top 10 predicitons
    #print(json.dumps(inception_v3.decode_predictions(np.array(pred['predictions']))[0]))