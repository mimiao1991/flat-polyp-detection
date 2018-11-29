from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import PIL.Image as pil
from glob import glob
import cv2
import matplotlib.pyplot as plt


import tensorflow.contrib.slim.nets

from imageselect_Dataloader import DataLoader
import os

from nets import *


flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("output_dir", "", "Dataset directory")
flags.DEFINE_integer("image_height", 540, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 675, "The size of of a sample batch")

FLAGS = flags.FLAGS

def main(_):
	FLAGS.checkpoint_dir='/playpen/code/Desktop/removereflection/spec_rm/checkpoints/'
	#FLAGS.dataset_dir = '/media/mimiao/5B90-5C09/polyps/'
	FLAGS.output_dir = '/home/mimiao/Desktop/haha/'
	with tf.Graph().as_default():
		#Load image and label
		x = tf.placeholder(shape=[None, 224*3, 224*3, 3], dtype=tf.float32)
		img_list = sorted(glob('/home/mimiao/Desktop/neg/*p*'))
		# # Define the model:
		#with tf.name_scope("Prediction"):
		with tf.name_scope("depth_prediction"):
			pred_disp, depth_net_endpoints = disp_net(x, is_training=False)
			saver = tf.train.Saver([var for var in tf.model_variables()])
			checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
		with tf.Session() as sess:
			saver.restore(sess, checkpoint)
			for i in range(len(img_list)):
			
				fh = open(img_list[i],'r')
				I = pil.open(fh)
				I = np.array(I)
				I = cv2.resize(I,(224*3,224*3),interpolation = cv2.INTER_AREA)
			
				I = I/255.0-0.5


				pred = sess.run(pred_disp,feed_dict={x:I[None,:,:,:]})

			
				z=cv2.resize(pred[0][0,:,:,:],(FLAGS.image_width,FLAGS.image_height),interpolation = cv2.INTER_CUBIC)
				#z = cv2.bilateralFilter(z,9,75,75)
				z = (z+0.5)*255
				z[z>255]=255

				#print np.shape(z)
				#import pdb;pdb.set_trace()
				#z=1.0/z#[0][0,:,:,0]
				#import pdb;pdb.set_trace()
				z = z.astype(np.uint8)
				#cv2.imwrite(FLAGS.output_dir+img_list[i].split('/')[-1]+'.jpg',z[:,:,0])

				#plt.show()
				if os.path.exists(FLAGS.output_dir):
					cv2.imwrite(FLAGS.output_dir+img_list[i].split('/')[-1],cv2.cvtColor(z, cv2.COLOR_RGB2BGR))
				else:
					os.mkdir(FLAGS.output_dir)
					cv2.imwrite(FLAGS.output_dir+img_list[i].split('/')[-1],cv2.cvtColor(z, cv2.COLOR_RGB2BGR))
				#z.save(FLAGS.output_dir+img_list[i].split('/')[-1]+'.jpg')
				#z.astype('int8').tofile(img_list[i]+'.jpg')
			
				print("The %dth frame is processed"%(i))



if __name__ == '__main__':
   tf.app.run()
