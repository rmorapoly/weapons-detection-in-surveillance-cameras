# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:04:55 2020

@author: 28771
"""
import time
start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import label_map_util 
 
import cv2
 
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
 
if StrictVersion(tf.__version__) < StrictVersion('1.4.0'):
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
 
# from utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
 
# # Model preparation 
MODEL_NAME = 'knife_detection'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
project_path = os.getcwd()
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_CKPT = os.path.join(project_path, PATH_TO_CKPT)

print(PATH_TO_CKPT)

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'gun.pbtxt')
PATH_TO_LABELS = os.path.join(os.getcwd(), 'gun.pbtxt')
 
NUM_CLASSES = 6
 
def detect_in_video():
    # VideoWriter is the responsible of creating a copy of the video
    # used for the detections but with the detections overlays. Keep in
    # mind the frame size has to be the same as original video.
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('video1_output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 250, (200, 100))
    # out = cv2.VideoWriter('SubmachineGun.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 720))
    
    #Load a (frozen) Tensorflow model into memory.   
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef() 
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
            
    #Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
 
    #cv2.VideoCapture (): 0 is the default camera of the computer, 1 can change the source.
            #for example: direct access to surveillance cameras
    video_name = 'video8'
    video_format = '.mp4'
    cap = cv2.VideoCapture(video_name + video_format)
    out = cv2.VideoWriter(video_name + '_output' + video_format, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10.0, (int(cap.get(3)),int(cap.get(4))))
    # out = cv2.VideoWriter(video_name + '_output' + video_format, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(cap.get(3)),int(cap.get(4))))

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object
            # was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class
            # label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            
 
            frame_count = 0
            while(cap.isOpened()):
                # Read the frame and capture frame-by-frame
                ret, frame = cap.read()
                
                # print(frame, 'end')
                # Recolor the frame. By default, OpenCV uses BGR color space.

                frame_count += 1
                if ret == False:
                    print('Finished')
                    break
                else:
                    print('frame', frame_count)
                    detect_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
                    image_np_expanded = np.expand_dims(detect_frame, axis=0)
    
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores,
                            detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
    
                    # Visualization of the results of a detection.
                    # note: perform the detections using a higher threshold
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        detect_frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=.20)
    
                    #cv2.imshow('detect_output', detect_frame)
                    output_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_RGB2BGR)
                    # cv2.imshow('detect_output', output_rgb)
                    out.write(output_rgb)
                    
                
                    #The number in waitKey(1) represents the invalid time before waiting for the key input.
                    #The unit is milliseconds. During this time period, the key 'q' will not be recorded. 
                    #In other words, after the invalid time, it is detected whether the key 'q' is pressed in the time period of the last image display. 
                    #If not, it will jump out of the if statement and capture and display the next frame of image.
                    #The return value of cv2.waitkey (1) is more than 8 bits, but only the last 8 bits are actually valid.
                    # To avoid interference, the remaining position is 0 through the '&' operation
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            #when everything done , release the capture
            out.release()
            cap.release()
           # cv2.destroyAllWindows()
 
def main():
    detect_in_video()
 
if __name__ =='__main__':
    main()
    
end =  time.time()
print("Execution Time: ", end - start)