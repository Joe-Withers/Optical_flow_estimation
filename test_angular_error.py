import datasets as ds
import numpy as np
from v001.SIFN import SIFN
version = 'v001'

def angular_error(flow_gt,flow):
    [x,y,_] = flow.shape
    f = np.concatenate((flow, np.ones((x,y,1))),axis=2)
    f_gt = np.concatenate((flow_gt, np.ones((x,y,1))),axis=2)
    #arccos doesnt seem to handle edge case of all 1.0
    ae = np.arccos( (np.sum(f*f_gt, axis=2)) / ( np.sqrt(np.sum(f**2, axis=2)) * np.sqrt(np.sum(f_gt**2, axis=2)) ) )
    return ae

#dataset info
n_data = 100
im_width = 128
im_height = 128
im_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/data/SyntheticData-b_plain-o_textured-dof_2/test/'
#load data
X, y = ds.load_my_synthetic_images(im_path, im_width=im_width, im_height=im_height, n_data=n_data)

#feedback paths
load_model_path = './'+version+'/training/'
dump_path = 'D:/Joe/Documents/University/Year 4/ResearchProject/My Code/temp_analysis/'
#feedback paras
save_step = 30
show_step = 10
bool_show_stuff = True
#training info
batch_size = 32
#train model
flow_estimator = SIFN()
predicted_flows = flow_estimator.run_network_with_data(X, load_model_path, batch_size=batch_size, save_flow_im_path=dump_path)

GT_flows = ds.load_my_synthetic_flows(im_path, n_data=n_data)

assert len(GT_flows)==len(predicted_flows)
AE = [np.mean(angular_error(GT_flows[i,:,:,:],predicted_flows[i,:,:,:])) for i in range(len(predicted_flows))]

print('Avg angular error',np.mean(AE))
