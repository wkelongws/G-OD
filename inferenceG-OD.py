import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("/home/shuo/Shuo/tensorflow_models/research")
sys.path.append("/home/shuo/Shuo/tensorflow_models/research/object_detection")

from utils import label_map_util

from utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def main():

	# What model to use.

	MODEL_ZOO = [
	            'faster_rcnn_inception_resnet_v2_atrous_coco',                   # 0  .37
	            'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco',      # 1  
	            'faster_rcnn_inception_v2_coco',                                 # 2  .28
	            'faster_rcnn_nas_coco',                                          # 3  .43
	            'faster_rcnn_nas_lowproposals_coco',                             # 4  
	            'faster_rcnn_resnet50_coco',                                     # 5  .30
	            'faster_rcnn_resnet50_lowproposals_coco',                        # 6  
	            'faster_rcnn_resnet101_coco',                                    # 7  .32
	            'faster_rcnn_resnet101_kitti',                                   # 8  .87
	            'faster_rcnn_resnet101_lowproposals_coco',                       # 9  
	            'rfcn_resnet101_coco',                                           # 10 .30
	            'ssd_inception_v2_coco',                                         # 11 .24
	            'ssd_mobilenet_v1_coco'                                          # 12 .21
	            ]                             

	USE_MODEL_ID = 3

	MODEL_NAME = MODEL_ZOO[USE_MODEL_ID]

	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	PATH_TO_CKPT = '/home/shuo/Shuo/tensorflow_models/research/object_detection/off_the_shelf_models/' + MODEL_NAME + '/frozen_inference_graph.pb'

	# List of the strings that is used to add correct label for each box. 
	PATH_TO_LABELS = os.path.join('/home/shuo/Shuo/tensorflow_models/research/object_detection/data', 'aic_label_map.pbtxt')
	# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
	NUM_CLASSES = 90
	if MODEL_NAME[-5:] == 'kitti':
	    PATH_TO_LABELS = os.path.join('data', 'kitti_label_map.pbtxt')
	    NUM_CLASSES = 2

	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')

	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)





	# For the sake of simplicity we will use only 2 images:
	# image1.jpg
	# image2.jpg
	# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
	# PATH_TO_TEST_IMAGES_DIR = 'test_images'
	# PATH_TO_TEST_IMAGES_DIR = '/home/shuo/Deformable-ConvNets/data/data_2/VOC1080/JPEGImages'
	PATH_TO_TEST_IMAGES_DIR = '/media/shuo/Shuo_NVIDIA/Dataset_for_LSTM/Frames/'

	# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
	# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(926,930) ]
	TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}'.format(i)) for i in np.sort(os.listdir(PATH_TO_TEST_IMAGES_DIR)) ]
	TEST_IMAGE_PATHS = TEST_IMAGE_PATHS[147302:]
	# Size, in inches, of the output images.
	IMAGE_SIZE = (15, 12)



	with detection_graph.as_default():
	  with tf.Session(graph=detection_graph) as sess:
	    # Definite input and output Tensors for detection_graph
	    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	    # Each box represents a part of the image where a particular object was detected.
	    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	    # Each score represent how level of confidence for each of the objects.
	    # Score is shown on the result image, together with the class label.
	    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	    # for i in range(len(os.listdir(PATH_TO_TEST_IMAGES_DIR))): #len(os.listdir(PATH_TO_TEST_IMAGES_DIR))
	    i = 147301
	    for image_path in TEST_IMAGE_PATHS:
	      i += 1
	      # image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i))
	      image = Image.open(image_path)
	      image_name = image_path.split('Frames/')[1].split('.')[0]
	      # the array based representation of the image will be used later in order to prepare the
	      # result image with boxes and labels on it.
	      image_np = load_image_into_numpy_array(image)
	      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	      image_np_expanded = np.expand_dims(image_np, axis=0)
	      # Actual detection.
	      (boxes, scores, classes, num) = sess.run(
	          [detection_boxes, detection_scores, detection_classes, num_detections],
	          feed_dict={image_tensor: image_np_expanded})
	      # Visualization of the results of a detection.
	      vis_util.visualize_and_output_boxes_and_labels_on_image_array(
	          image_np,
	          np.squeeze(boxes),
	          np.squeeze(classes).astype(np.int32),
	          np.squeeze(scores),
	          category_index,
	          output_path = '/media/shuo/Shuo_NVIDIA/Dataset_for_LSTM/Annotations_continue/',
	          image_name = '{}'.format(image_name),
	          use_normalized_coordinates=True,
	          max_boxes_to_draw=20,
	          min_score_thresh=.05,
	          agnostic_mode=False,
	          line_thickness=8)
	      print('{}/{}'.format(i,len(os.listdir(PATH_TO_TEST_IMAGES_DIR))))
	#       plt.figure(figsize=IMAGE_SIZE)
	#       plt.imshow(image_np)



if __name__ == '__main__':
	main()