import numpy as np
import cv2

def normalise_image(im):
    im = np.array(im)
    return im / 255.

#***************************my synthetic data**********************************
def load_my_synthetic_images(im_path, n_data=100):
    i=0
    f_im1 = im_path+'images/'+str(i)+'-I1.png'
    f_im2 = im_path+'images/'+str(i)+'-I2.png'
    im1 = normalise_image(cv2.imread(f_im1,0))
    im2 = normalise_image(cv2.imread(f_im2,0))
    [width, height] = im1.shape
    X = np.zeros((n_data, width, height, 2))
    X[i,:,:,0] = im1
    X[i,:,:,1] = im2

    for i in range(1,n_data):
        f_im1 = im_path+'images/'+str(i)+'-I1.png'
        f_im2 = im_path+'images/'+str(i)+'-I2.png'
        im1 = normalise_image(cv2.imread(f_im1,0))
        im2 = normalise_image(cv2.imread(f_im2,0))
        X[i,:,:,0] = im1
        X[i,:,:,1] = im2
    return X

def load_my_synthetic_image_filenames(im_path, n_data=100):
    f_im1s = [im_path+'images/'+str(i)+'-I1.png' for i in range(0,n_data)]
    f_im2s = [im_path+'images/'+str(i)+'-I2.png' for i in range(0,n_data)]
    return f_im1s, f_im2s

def load_my_synthetic_flows(im_path, n_data=100):
    i=0
    f_flow = im_path+'flow/'+str(i)+'-I1-I2.npy'
    flow = np.load(f_flow)
    [width, height, _] = flow.shape
    flows = np.zeros((n_data, width, height, 2))
    flows[i,:,:,:] = flow
    for i in range(1,n_data):
        f_flow = im_path+'flow/'+str(i)+'-I1-I2.npy'
        flows[i,:,:,:] = np.load(f_flow)
    return 0.5*flows

def load_my_synthetic_flows_filenames(im_path, n_data=100):
    f_flow = [im_path+'flow/'+str(i)+'-I1-I2.npy' for i in range(0,n_data)]
    return f_flow

#***************************flying chairs data**********************************
def load_synthetic_chairs_images(im_path, n_data=100):
    i=0
    f_im1 = im_path+('0'*(7-len(str(i))))+str(i)+'-img_0.png'
    f_im2 = im_path+('0'*(7-len(str(i))))+str(i)+'-img_1.png'
    im1 = normalise_image(cv2.imread(f_im1,0))
    im2 = normalise_image(cv2.imread(f_im2,0))
    [width, height] = im1.shape
    X = np.zeros((n_data, width, height, 2))
    X[i,:,:,0] = im1
    X[i,:,:,1] = im2

    for i in range(1,n_data):
        f_im1 = im_path+('0'*(7-len(str(i))))+str(i)+'-img_0.png'
        f_im2 = im_path+('0'*(7-len(str(i))))+str(i)+'-img_1.png'
        im1 = normalise_image(cv2.imread(f_im1,0))
        im2 = normalise_image(cv2.imread(f_im2,0))
        X[i,:,:,0] = im1
        X[i,:,:,1] = im2
    return X

def load_synthetic_chairs_image_filenames(im_path, n_data=100):
    f_im1s = [im_path+('0'*(7-len(str(i))))+str(i)+'-img_0.png' for i in range(0,n_data)]
    f_im2s = [im_path+('0'*(7-len(str(i))))+str(i)+'-img_1.png' for i in range(0,n_data)]
    return f_im1s, f_im2s

def load_synthetic_chairs_flows(im_path, n_data=100):
    i=0
    f_flow = im_path+('0'*(7-len(str(i))))+str(i)+'-flow_01.flo'
    f = open(f_flow, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    flows = np.zeros((n_data, height, width, 2))
    flows_t = np.zeros_like(flows)
    flows[i,:,:,:] = flow.astype(np.float32)
    for i in range(1,n_data):
        f_flow = im_path+('0'*(7-len(str(i))))+str(i)+'-flow_01.flo'
        f = open(f_flow, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

        flows[i,:,:,:] = flow.astype(np.float32)
    flows_t[:,:,:,0] = flows[:,:,:,1]
    flows_t[:,:,:,1] = flows[:,:,:,0]
    return flows_t

def load_synthetic_chairs_flows_filenames(im_path, n_data=100):
    f_flows = [im_path+('0'*(7-len(str(i))))+str(i)+'-flow_01.flo' for i in range(0,n_data)]
    return f_flows
