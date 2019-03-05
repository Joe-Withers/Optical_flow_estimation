import datasets as ds
from v001.SIFN import SIFN
version = 'v001'

#dataset info
n_data = 639
im_width = 384
im_height = 512
# im_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/data/SyntheticData-b_plain-o_textured-dof_2/test/'
im_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/data/FlyingChairs2/FlyingChairs2/FlyingChairs2/val/'
#load data
# im1_filenames,im2_filenames,y_filenames = ds.load_my_synthetic_image_filenames(im_path, n_data=n_data)
im1_filenames,im2_filenames,y_filenames = ds.load_synthetic_chairs_image_filenames(im_path, n_data=n_data)
#feedback paths
load_model_path = './'+version+'/training/'
dump_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/temp_analysis/'
#training info
batch_size = 4
#train model
flow_estimator = SIFN()
flow_estimator.run_network(im1_filenames, im2_filenames, im_width, im_height, load_model_path, batch_size = 4, save_flow_im_path = dump_path)
