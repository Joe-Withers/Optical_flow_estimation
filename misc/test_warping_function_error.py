"""given GT flow and 2 images how accurate is the warp function"""
import sys
sys.path.append('./')
import numpy as np
import tensorflow as tf
import cv2
version = 'v002'

if version=='v001':
    import v001.datasets as ds
    from v001.UnSupFlowNet import warping_error, visualise_flow

if version=='v002':
    import v002.datasets as ds
    from v002.UnSupFlowNet import warping_error, visualise_flow


#dataset info
# n_datas = [100,100,100,100]
# im_paths = ['./data/full_texture/test/','./data/obj_texture/test/','./data/no_texture/test/','./data/occlusion/test/']
# prefixes = ['full_texture','obj_texture','no_texture','occlusion']
n_datas = [1000]
im_paths = ['./data/FlyingChairs2/train/']
prefixes = ['flying_chairs']
dump_path = './temp_analysis/'
for im_path, prefix, n_data in zip(im_paths, prefixes, n_datas):
    #load data
    # X = ds.load_my_synthetic_images(im_path, n_data=n_data)
    # GT_flows = ds.load_my_synthetic_flows(im_path, n_data=n_data)
    X = ds.load_synthetic_chairs_images(im_path, n_data=n_data)
    GT_flows = ds.load_synthetic_chairs_flows(im_path, n_data=n_data)

    [_, im_width, im_height, _] = X.shape
    [_, f_width, f_height, _] = GT_flows.shape
    print(X.shape)
    print(GT_flows.shape)
    assert im_width==f_width and im_height==f_height
    im1 = tf.placeholder('float', [None, im_width, im_height])
    im2 = tf.placeholder('float', [None, im_width, im_height])
    flowt = tf.placeholder('float', [None, f_width, f_height, 2])
    warping_err, warped_im = warping_error(im2, im1, -flowt)
    with tf.Session() as sess:
        WE, WI = sess.run([warping_err, warped_im], feed_dict = {im1: X[:,:,:,0], im2: X[:,:,:,1], flowt: GT_flows})

    for i in range(0,10):
        image1 = X[i,:,:,0]
        image2 = X[i,:,:,1]
        w_image1 = WI[i,:,:,:]
        print(image1.shape)
        print(image2.shape)
        print(w_image1.shape)
        cv2.imwrite(dump_path+prefix+str(i)+'im_1.png', np.multiply(image1,255))
        cv2.imwrite(dump_path+prefix+str(i)+'im_2.png', np.multiply(image2,255))
        cv2.imwrite(dump_path+prefix+str(i)+'flow.png', visualise_flow(GT_flows[i,:,:,:]))
        cv2.imwrite(dump_path+prefix+str(i)+'warped_im.png', np.multiply(w_image1,255))
    print(prefix,': Avg warping function error',WE)
