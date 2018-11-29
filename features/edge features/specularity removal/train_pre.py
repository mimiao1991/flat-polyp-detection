from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np

import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.python.slim.learning import train_step

from imageselect_Dataloader import DataLoader
import os

from nets import *

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("validate_dir", "./validation", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("batch_size", 5, "The size of of a sample batch")
flags.DEFINE_integer("max_steps", 100000, "Maximum number of training iterations")
flags.DEFINE_string("pretrain_weight_dir", "./pretrained", "Directory name to pretrained weights")
flags.DEFINE_integer("validation_check", 100, "Directory name to pretrained weights")


flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")

flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")


FLAGS = flags.FLAGS

FLAGS.num_scales = 4
FLAGS.smooth_weight = 2.0


slim = tf.contrib.slim
resnet_v2 = tf.contrib.slim.nets.resnet_v2

def compute_smooth_loss(pred_disp):
	def gradient(pred):
		D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
		D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
		return D_dx, D_dy
	dx, dy = gradient(pred_disp)
	dx2, dxdy = gradient(dx)
	dydx, dy2 = gradient(dy)
	smoothout = (tf.reduce_mean(tf.abs(dx2)) + tf.reduce_mean(tf.abs(dxdy)) + tf.reduce_mean(tf.abs(dydx)) + tf.reduce_mean(tf.abs(dy2)))
	return smoothout


def main(_):

	if not tf.gfile.Exists(FLAGS.checkpoint_dir):
	  tf.gfile.MakeDirs(FLAGS.checkpoint_dir)



	with tf.Graph().as_default():
		

		#============================================
		#Load image and label
		#============================================
		with tf.name_scope("data_loading"):
			imageloader = DataLoader(FLAGS.dataset_dir,
									 FLAGS.batch_size,
									 FLAGS.image_height, 
									 FLAGS.image_width,
									 'train')
			#import pdb;pdb.set_trace()
			# For depth prediction I resized every image to [224, 224].
			# For spec removal you can either use this setting or remove the resize part both 
			# in data loading and every places where 224 appear.			
			image,label = imageloader.load_train_batch()


			
		#============================================
		#Define the model
		#============================================
		with tf.name_scope("depth_prediction"):

	        	global_step = tf.Variable(0, 
	                                       name='global_step', 
	                                       trainable=False)
	        	incr_global_step = tf.assign(global_step, 
	                                          global_step+1)


			pred_disp, depth_net_endpoints = disp_net(image, 
			                                      is_training=True)

			#import pdb;pdb.set_trace()
		#============================================
		#Validation
		#First make the code running, then I will 
		#add a validation here
		#============================================

		

		#============================================	
		#Specify the loss function:
		#============================================

		with tf.name_scope("compute_loss"):
			pixel_loss = 0

			#Probably dont need smooth loss for spec remove
			smooth_loss = 0

			for s in range(FLAGS.num_scales):


				# smooth_loss += FLAGS.smooth_weight/(2**s) * \
				#     compute_smooth_loss(pred_disp[s])


				# For depth prediction I resized every image to [224, 224].
				# For spec removal you can either use this setting or remove the resize part both 
				# in data loading and every places where 224 appear.
				curr_label = tf.image.resize_area(label, 
					[int(224*3/(2**s)), int(224*3/(2**s))])                

				# Mean squared error between prediction and groundtruth
				curr_depth_error = tf.square(curr_label - pred_disp[s])
				pixel_loss += tf.reduce_mean(curr_depth_error)

			total_loss = pixel_loss + smooth_loss 




		#============================================
		#Start training
		#============================================

		with tf.name_scope("train_op"):
	
		    tf.summary.scalar('losses/total_loss', total_loss)

	            tf.summary.image('scale%d_pred_removal' % s, \
	                     pred_disp[0])
	            tf.summary.image('scale%d_pred_removal' % s, \
	                     image)
	        


		    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                           10000, 0.9, staircase=True)

			# Specify the optimization scheme:
		    optimizer = tf.train.AdamOptimizer(learning_rate,FLAGS.beta1)

			# create_train_op that ensures that when we evaluate it to get the loss,
			# the update_ops are done and the gradient updates are computed.
		    train_op = slim.learning.create_train_op(total_loss, optimizer)

		    saver = tf.train.Saver([var for var in tf.model_variables()])


		with tf.Session() as sess:
		    merged = tf.summary.merge_all()
		    train_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/sum',
		                                          sess.graph)
		    tf.initialize_all_variables().run()
		    tf.initialize_local_variables().run()

		    coord = tf.train.Coordinator()
		    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		    

		    #Continue training
		    if FLAGS.continue_train:
		        if FLAGS.init_checkpoint_file is None:
		            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
		        else:
		            checkpoint = FLAGS.init_checkpoint_file
		        print("Resume training from previous checkpoint: %s" % checkpoint)
		        saver.restore(sess, checkpoint)

		    for step in range(1, FLAGS.max_steps):
		        #print("steps %d" % (step))
		        fetches = {
		            "train": train_op,
		            "global_step": global_step,
		            "incr_global_step": incr_global_step
		        }

		        if step % FLAGS.summary_freq == 0:
		            fetches["loss"] = total_loss
		            fetches["summary"] = merged


		        results = sess.run(fetches)
		        gs = results["global_step"]

		        if step % FLAGS.summary_freq == 0:
		            train_writer.add_summary(results["summary"], gs)

		            print("steps: %d === loss: %.3f" \
		                    % (gs,
		                        results["loss"]))



		        if step % FLAGS.save_latest_freq == 0:
		            saver.save(sess, FLAGS.checkpoint_dir+'/model', global_step=gs)

		    coord.request_stop()
		    coord.join(threads)







if __name__ == '__main__':
   tf.app.run()
