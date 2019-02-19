import numpy as np
import cv2

def normalise_image(im):
    im = np.array(im)
    return im / 255

def load_my_synthetic_images(im_path, im_width=128, im_height=128, n_data=100):
    X = np.zeros((n_data, im_width, im_height, 2))
    y = np.zeros((n_data, im_width, im_height, 1))
    for i in range(0,n_data):
        f_im1 = im_path+'images/'+str(i)+'-I1.png'
        f_im2 = im_path+'images/'+str(i)+'-I2.png'
        f_im3 = im_path+'images/'+str(i)+'-I2.png'

        im1 = normalise_image(cv2.resize(cv2.imread(f_im1,0),(im_width, im_height)))
        im2 = normalise_image(cv2.resize(cv2.imread(f_im2,0),(im_width, im_height)))
        im3 = normalise_image(cv2.resize(cv2.imread(f_im3,0),(im_width, im_height)))

        X[i,:,:,0] = im1
        X[i,:,:,1] = im3
        y[i,:,:,0] = im2

    return X, y

def load_my_synthetic_flows(im_path, im_width=128, im_height=128, n_data=100):
    flows = np.zeros((n_data, im_width, im_height, 2))
    for i in range(0,n_data):
        f_flow = im_path+'flow/'+str(i)+'-I1-I2.npy'
        flows[i,:,:,:] = np.load(f_flow)

    return flows

def load_synthetic_chairs(im_path, im_width=128, im_height=128, n_data=100):
    train_x = np.zeros((n_data, im_width, im_height, 2))
    train_y = np.zeros((n_data, im_width, im_height, 1))
    for i in range(0,n_data):
        f_im1 = im_path+('0'*(7-len(str(i))))+str(i)+'-img_0.png'
        f_im2 = im_path+('0'*(7-len(str(i))))+str(i)+'-img_1.png'
        f_im3 = im_path+('0'*(7-len(str(i))))+str(i)+'-img_1.png'

        im1 = normalise_image(cv2.resize(cv2.imread(f_im1,0),(im_width, im_height)))
        im2 = normalise_image(cv2.resize(cv2.imread(f_im2,0),(im_width, im_height)))
        im3 = normalise_image(cv2.resize(cv2.imread(f_im3,0),(im_width, im_height)))

        X[i,:,:,0] = im1
        X[i,:,:,1] = im3
        y[i,:,:,0] = im2

    return X, y
