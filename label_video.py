# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 23:54:28 2019

@author: owner
useful for identifying classifying images using webcam on your laptop
"""
# =============================================================================
# 
# from imutils.video import VideoStream
# from imutils.video import FPS
# =============================================================================
import os
import cv2
import tensorflow as tf

def orange13():
    cap = cv2.VideoCapture(0)
    i=0
    label_pointer = [line.rstrip() for line
                   in tf.io.gfile.GFile("C:/Spyder_projects/orange13/retrained_labels.txt")]
    
    with tf.io.gfile.GFile("C:/Spyder_projects/orange13/retrained_graph.pb", 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    while True:
        
        success, image = cap.read()
        image_name = 'image{}.jpg'.format(i)
        cv2.imwrite(os.path.join('C:\Spyder_projects\orange13\capture_images',image_name),image)
        image_data = tf.io.gfile.GFile(os.path.join('C:\Spyder_projects\orange13\capture_images',image_name), 'rb').read()# Loads label file, strips off carriage return
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if i >= 51:
            i=0
        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, 
                                   {'DecodeJpeg/contents:0': image_data})           	
        print('No. of Iterations: {}'.format(i+1))
        i+=1
        desc_index = predictions[0].argsort()[-len(predictions[0]):][::-1]
#        print("Predictions: ",predictions,"\n")
#        print("TOP_K: ",top_k)
        for index in desc_index:
            label_string = label_pointer[index]
            score = predictions[0][index]
            print('%s (score = %.5f)' % (label_string, score))
        
        
    cap.release()
    cv2.destroyAllWindows()
    
orange13()
# =============================================================================
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile("C:/Spyder_projects/orange13/retrained_graph.pb", 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')
#     
# with detection_graph.as_default():
#   with tf.Session(graph=detection_graph) as sess:
#       while True:
#           ret, image_np = cap.read()
#           # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#           image_np_expanded = np.expand_dims(image_np, axis=0)
#           image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#           # Each box represents a part of the image where a particular object was detected.
#         softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#         predictions = sess.run(softmax_tensor, 
#                                {image_tensor: image_np_expanded})
#         # Sort to show labels of first prediction in order of confidence
#         top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#         for node_id in top_k:
#             human_string = label_lines[node_id]
#             score = predictions[0][node_id]
#             print('%s (score = %.5f)' % (human_string, score))
#             
#           cv2.imshow('image classification', cv2.resize(image_np, (800,600)))
#           if cv2.waitKey(25) 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break
# =============================================================================
#def video_init():

#def label_video():
#    import tensorflow as tf,sys
#    while True:
#        frame = vs.read()
#        frame = imutils.resize(frame,width=500)
#        image_data = frame
#        label_lines = [line.rstrip() for line
#                       in tf.io.gfile.GFile("C:/Spyder_projects/orange13/retrained_labels.txt")]# Unpersists graph from file
#        with tf.io.gfile.GFile("C:/Spyder_projects/orange13/retrained_graph.pb", 'rb') as f:
#            graph_def = tf.compat.v1.GraphDef()
#            graph_def.ParseFromString(f.read())
#            _ = tf.import_graph_def(graph_def, name='')# Feed the image_data as input to the graph and get first prediction
#        with tf.compat.v1.Session() as sess:
#            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#            predictions = sess.run(softmax_tensor, 
#                                   {'DecodeJpeg/contents:0': image_data})
#            # Sort to show labels of first prediction in order of confidence
#            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#            for node_id in top_k:
#                human_string = label_lines[node_id]
#                score = predictions[0][node_id]
#                print('%s (score = %.5f)' % (human_string, score))
#        if key == ord("q"):
#            break
        
# =============================================================================
# except Exception as ex:
#     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
#     message = template.format(type(ex).__name__, ex.args)
#     print(message)
# =============================================================================
    
        
    