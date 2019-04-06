import datasets as ds
import cv2
import numpy as np
def visualise_flow(flow):
    [width, height, _] = flow.shape
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((width, height, 3))
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype('uint8'),cv2.COLOR_HSV2BGR)
    return bgr


n_data = 50
im_path = './data/FlyingChairs2/train/'
save_flow_im_path = './data/FlyingChairs2/train/images/'

X = ds.load_synthetic_chairs_images(im_path, n_data=n_data)
GT_flows = ds.load_synthetic_chairs_flows(im_path, n_data=n_data)

(hm_data,_,_,_) = GT_flows.shape
for idx in range(0,hm_data):
    if show_stuff = True:
        cv2.imshow( "im_1", cv2.resize(X[1,:,:,0],(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.imshow( "im_2", cv2.resize(X[1,:,:,1],(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.imshow( "flow_10", cv2.resize(visualise_flow(GT_flows[1,:,:,:]),(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(save_flow_im_path+str(idx)+'.png',
        cv2.resize(visualise_flow(GT_flows[idx,:,:,:]), (300,300), interpolation = cv2.INTER_NEAREST))
