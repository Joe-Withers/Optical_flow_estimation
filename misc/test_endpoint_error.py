import datasets as ds
import numpy as np
from v001.SIFN import SIFN
version = 'v001'

#dataset info
n_data = 100
im_width = 128
im_height = 128
# im_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/data/SyntheticData-b_plain-o_textured-dof_2/test/'
im_path = './data/obj_texture/test/'
#load data
X, y = ds.load_my_synthetic_images(im_path, im_width=im_width, im_height=im_height, n_data=n_data)
# X, y = ds.load_synthetic_chairs_imags(im_path, n_data=n_data)


#feedback paths
load_model_path = './'+version+'/training/obj_texture_sup/'
dump_path = 'ignore'
#feedback paras
save_step = 30
show_step = 10
bool_show_stuff = False
#training info
batch_size = 32
#train model
flow_estimator = SIFN()
predicted_flows = flow_estimator.run_network_with_data(X, load_model_path, batch_size=batch_size, save_flow_im_path=dump_path)

GT_flows = ds.load_my_synthetic_flows(im_path, n_data=n_data)

assert len(GT_flows)==len(predicted_flows)
EE = [np.mean(np.sqrt(GT_flows[i,:,:,:]**2+predicted_flows[i,:,:,:]**2)) for i in range(len(predicted_flows))]


print('Avg endpoint error',np.mean(EE))
