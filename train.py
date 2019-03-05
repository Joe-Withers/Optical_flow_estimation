import datasets as ds
from v001.SIFN import SIFN
version = 'v001'

#dataset info
n_data = 10
im_width = 384
im_height = 512
# im_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/data/SyntheticData-b_plain-o_textured-dof_2/train/'
im_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/data/FlyingChairs2/FlyingChairs2/FlyingChairs2/train/'
#load data
# im1_filenames,im2_filenames,y_filenames = ds.load_my_synthetic_image_filenames(im_path, n_data=n_data)
im1_filenames,im2_filenames,y_filenames = ds.load_synthetic_chairs_image_filenames(im_path, n_data=n_data)

#feedback paths
load_model_path = './'+version+'/training/'
save_model_path = './'+version+'/training/'
dump_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/temp_analysis/'
#feedback paras
save_step = 1
show_step = 10
bool_show_stuff = False
#training info
batch_size = 4
n_epochs = 10000000
#hyperparameters
epsilon = 0.005
lambda1 = 0.00000005
#train model
flow_estimator = SIFN()
flow_estimator.train_network(im1_filenames, im2_filenames, im_width, im_height, t=1, hm_epochs=n_epochs, epsilon=epsilon, lambda1=lambda1, batch_size=batch_size,
    save_flow_im_path='ignore', save_step=save_step, show_step=show_step, bool_show_stuff=bool_show_stuff,
    load_model_path=load_model_path, save_model_path=save_model_path)
