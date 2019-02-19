import datasets as ds
from SIFN_102.SIFN import SIFN

#dataset info
n_data = 100
im_width = 128
im_height = 128
im_path = './data/SyntheticData-b_plain-o_textured-dof_2/test/images/'
#load data
X, y = ds.load_my_synthetic(im_path, im_width=im_width, im_height=im_height, n_data=n_data)

#feedback paths
load_model_path = './SIFN_102/training/'
save_model_path = './SIFN_102/training/'
dump_path = './temp_analysis/'
#feedback paras
save_step = 30
show_step = 10
bool_show_stuff = True
#training info
batch_size = 32
n_epochs = 10000000
#hyperparameters
epsilon = 0.005
lambda1 = 0.00000005
#train model
flow_estimator = SIFN()
flow_estimator.train_network(X, y, t=1, hm_epochs=n_epochs, epsilon=epsilon, lambda1=lambda1, batch_size=batch_size,
    save_flow_im_path=dump_path, save_step=save_step, show_step=show_step, bool_show_stuff=bool_show_stuff,
    load_model_path=model_path, save_model_path=save_model_path)
