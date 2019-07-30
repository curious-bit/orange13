"""
This code is able to classify oranges and apple in live video feed.
Graph has been retrained on Inception pre-trained model from Google using
Transfer Learning Method 
"""
# =============================================================================
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

    
