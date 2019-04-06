import datasets as ds
from v001.SIFN import SIFN
version = 'v001'

#dataset info
n_data = 100
im_width = 128
im_height = 128
im_path = './data/affine/test/'
#load data
im1_filenames,im2_filenames = ds.load_my_synthetic_image_filenames(im_path, n_data=n_data)
# im1_filenames,im2_filenames = ds.load_synthetic_chairs_image_filenames(im_path, n_data=n_data)
#feedback paths
load_model_path = './'+version+'/training/kaggle/affine_sup_5e-8_16/'
dump_path = './temp_analysis/'
#training info
batch_size = 4
#train model
flow_estimator = SIFN()
flow_estimator.run_network(im1_filenames, im2_filenames, im_width, im_height, load_model_path, batch_size = 4, save_flow_im_path = dump_path)
