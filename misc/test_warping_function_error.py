"""given GT flow and 2 images how accurate is the warp function"""
import datasets as ds
import numpy as np
import tensorflow as tf
import cv2
from v001.SIFN import warping_error, visualise_flow
version = 'v001'

#dataset info
n_data = 100
im_width = 128
im_height = 128
im_path = './data/obj_texture/test/'
#load data
X = ds.load_my_synthetic_images(im_path, im_width=im_width, im_height=im_height, n_data=n_data)
GT_flows = ds.load_my_synthetic_flows(im_path, n_data=n_data)

im1 = tf.placeholder('float', [None, im_width, im_height])
im2 = tf.placeholder('float', [None, im_width, im_height])
flowt = tf.placeholder('float', [None, im_width, im_height, 2])
warping_err, warped_im = warping_error(im2, im1, flowt, 1)
with tf.Session() as sess:
    WE, WI = sess.run([warping_err, warped_im], feed_dict = {im1: X[:,:,:,0], im2: X[:,:,:,1], flowt: -GT_flows})

i = 0
cv2.imshow('im 1', cv2.resize(X[i,:,:,0],(300,300)))
cv2.imshow('im 2', cv2.resize(X[i,:,:,1],(300,300)))
cv2.imshow('flow', cv2.resize(visualise_flow(-GT_flows[i,:,:,:]),(300,300)))
cv2.imshow('warped im', cv2.resize(WI[i,:,:,:],(300,300)))
cv2.waitKey(0)
print('Avg warping function error',WE)
