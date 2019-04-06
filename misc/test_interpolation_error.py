import datasets as ds
import numpy as np
from v001.SIFN import SIFN
version = 'v001'
#***************************************TODO: check the warping logic (wrong way?)*****************************************************************************
#dataset info
n_data = 100
im_width = 128
im_height = 128
im_path = './data/obj_texture/test/'
#load data
X, y = ds.load_my_synthetic_images(im_path, im_width=im_width, im_height=im_height, n_data=n_data)

#feedback paths
load_model_path = './'+version+'/training/obj_texture_sup/'
dump_path = 'ignore'
#feedback paras
save_step = 30
show_step = 10
bool_show_stuff = True
#training info
batch_size = 32
#train model
flow_estimator = SIFN()
predicted_flows = flow_estimator.run_network_with_data(X, load_model_path, batch_size=batch_size, save_flow_im_path=dump_path)

def bilinear_interpolate(im, flow):
    """ takes a single image and a flow for that image """
    [x,y,_] = flow.shape
    idxs = np.indices((x,y))
    coord = np.swapaxes(flow,0,2) + idxs
    x = np.asarray(coord[0])
    y = np.asarray(coord[1])

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ x0, y0]
    Ib = im[ x0, y1]
    Ic = im[ x1, y0]
    Id = im[ x1, y1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (1-(y1-y))
    wc = (1-(x1-x)) * (y1-y)
    wd = (1-(x1-x)) * (1-(y1-y))

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

assert len(X)==len(predicted_flows)
warped_ims = np.array([bilinear_interpolate(X[i,:,:,1], predicted_flows[i,:,:,:]) for i in range(len(X))])

assert len(y)==len(warped_ims)
IE = [np.mean(np.sqrt(y[i,:,:,0]**2+warped_ims[i,:,:]**2)) for i in range(len(y))]

print('Avg interpolation error',np.mean(IE))
