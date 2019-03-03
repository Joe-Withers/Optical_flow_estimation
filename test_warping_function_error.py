"""given GT flow and 2 images how accurate is the warp function"""
import datasets as ds
import numpy as np
import tensorflow as tf
from v001.SIFN import warping_error
version = 'v001'

#dataset info
n_data = 100
im_width = 128
im_height = 128
im_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/data/SyntheticData-b_plain-o_textured-dof_2/test/'
#load data
X, y = ds.load_my_synthetic_images(im_path, im_width=im_width, im_height=im_height, n_data=n_data)
GT_flows = ds.load_my_synthetic_flows(im_path, n_data=n_data)

xt = tf.placeholder('float', [None, im_width, im_height])
yt = tf.placeholder('float', [None, im_width, im_height])
flowt = tf.placeholder('float', [None, im_width, im_height, 2])
warping_err, _ = warping_error(xt, yt, flowt, 1)
with tf.Session() as sess:
    WE = sess.run(warping_err, feed_dict = {xt: X[:,:,:,0], yt: y[:,:,:,0], flowt: GT_flows})

print('Avg warping function error',WE)
