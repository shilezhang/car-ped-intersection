"""
tracking use: https://gist.github.com/ManivannanMurugavel/85cab9e1549a0722cc06712ab92e2c02

created: Dec 14th
use mask rcnn + tracking to process video and output trajectory

"""
# from ctypes import *
from __future__ import division, print_function, absolute_import





from numpy.linalg import norm


import os
from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
import argparse
import random
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
model_filename = '/home/shilezhang/yolo-application/deep_sort_yolov3/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

warnings.filterwarnings('ignore')


# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco
from samples.coco import coco
import shutil
import glob
import pandas as pd
from keras import backend
import tensorflow as tf

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

# config = coco.CocoConfig()

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
# config.display()


warnings.filterwarnings('ignore')
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']





colors = [tuple(255 * np.random.rand(3)) for _ in range(15)]



detected_objects = ['person','car','truck','bus','bicycle', 'motorcycle']
font = cv2.FONT_HERSHEY_SIMPLEX


less = 100



winName = 'Mask-RCNN Object detection in OpenCV'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

def main(videopath):
    frame_index = -1
    fps= 0 
    a = videopath.split('.')[-2].split('/')[-1]

    filepath = './output_traj/' + a
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
    os.mkdir(filepath)

    video_capture = cv2.VideoCapture(videopath)

   
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_FourCC = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output{}.avi'.format(a),cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, (width,height))


    n_frame = 8
    ref_n_frame_axies = []
    ref_n_frame_label = []
    ref_n_frame_axies_flatten = []
    ref_n_frame_label_flatten = []
    label_cnt = 1

    min_distance = 50
    

    while(True):
        t1 = time.time()  
        ret, frame = video_capture.read()  
        # Create a 4D blob from a frame.
        frame_index += 1
        if ret == True:
            if frame_index%15 != 0:
                continue
            
            cur_frame_axies = []
            cur_frame_label = []
           
            results = model.detect([frame], verbose=1)
            r = results[0]
            outputs, masks, classIds  = r['rois'], r['masks'], r['class_ids']
            
            #posttprocess
           
            numDetections = classIds.shape[0]

            # for color,output in zip(colors,outputs):
            for color,i in zip(colors,range(numDetections)):

                # box = outputs[0, 0, i]
                output = outputs[i]
                top,left,bottom,right = outputs[i]
                # mask = masks[i]
                # score = box[2]
                #extract bounding box 
                classId = classIds[i]
                text = class_names[classId]
            


                lbl = float('nan')
                
                if text in detected_objects:
                    
                    text = 'ped' if text == 'person' else 'car'
                    min_distance = 30 if text == 'ped' else 50

                    if(len(ref_n_frame_label_flatten) > 0):
                        b = np.array([(left,top)])
                        a = np.array(ref_n_frame_axies_flatten)
                        distance = norm(a-b,axis=1)
                        min_value = distance.min()
                        
                        if(min_value < min_distance):
                            idx = np.where(distance==min_value)[0][0]
                            lbl = ref_n_frame_label_flatten[idx]
                            # print(idx)
                    if(math.isnan(lbl)):
                        lbl = label_cnt
                        label_cnt += 1
                    cur_frame_label.append(lbl)
                    cur_frame_axies.append((left,top))
                   
                    cv2.rectangle(frame,(left,top),(right,bottom),color,2)
                    cv2.putText(frame,'{}{}'.format(text,lbl),(left,top), font, 1,(255,255,255),2)

                    hello = filepath+'/tracking_'+text+str(lbl)+".txt"
            
                    with open(hello,'a+') as list_file:
                        list_file.write(str(frame_index)+' ')
                        list_file.write(str(left))
                        list_file.write(' ')
                        list_file.write(str(top))
                        list_file.write(' ')
                        list_file.write(str(right))
                        list_file.write(' ')
                        list_file.write(str(bottom))
                        list_file.write(' ')
                        list_file.write(str(fps))
                        list_file.write('\n')
                    list_file.close()


            if(len(ref_n_frame_axies) == n_frame):
                del ref_n_frame_axies[0]
                del ref_n_frame_label[0]
            ref_n_frame_label.append(cur_frame_label)
            ref_n_frame_axies.append(cur_frame_axies)
            ref_n_frame_axies_flatten = [a for ref_n_frame_axie in ref_n_frame_axies for a in ref_n_frame_axie]
            ref_n_frame_label_flatten = [b for ref_n_frame_lbl in ref_n_frame_label for b in ref_n_frame_lbl]
            t2 = time.time()  
            fps  = ( fps + (1./(t2-t1)) ) / 2
            
            cv2.imshow('image',frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    

    k = 0
    filepath = '/home/shilezhang/Downloads/keypoint_park/'
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    weights_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    model.load_weights(weights_path, by_name=True)

    
    
    for subfolder_name in os.listdir(filepath):
        for filename in glob.glob(filepath+'/'+subfolder_name+'/*.mov'):      
            k += 1
            main(videopath = filename)
            
            
            
    print(str(k))