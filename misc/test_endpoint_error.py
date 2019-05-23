import sys
sys.path.append('./')
import numpy as np
version = 'v002'

if version=='v001':
    import v001.datasets as ds
    from v001.UnSupFlowNet import UnSupFlowNet

if version=='v002':
    import v002.datasets as ds
    from v002.UnSupFlowNet import UnSupFlowNet

#dataset info
n_data = 1000
im_path = './data/no_texture/train/'
#load data
X = ds.load_my_synthetic_images(im_path, n_data=n_data)
GT_flows = ds.load_my_synthetic_flows(im_path, n_data=n_data)
# X = ds.load_synthetic_chairs_images(im_path, n_data=n_data)
# GT_flows = ds.load_synthetic_chairs_flows(im_path, n_data=n_data)

#feedback paths
load_model_path = './'+version+'/training/kaggle/no_texture/sup_5e-8/'
dump_path = 'ignore'
#feedback paras
save_step = 30
show_step = 10
bool_show_stuff = False
#training info
batch_size = 32
#train model
flow_estimator = UnSupFlowNet()
predicted_flows = flow_estimator.run_network_with_data(X, load_model_path, batch_size=batch_size, save_flow_im_path=dump_path)

assert len(GT_flows)==len(predicted_flows)
EE = [np.mean(np.sqrt(GT_flows[i,:,:,:]**2+predicted_flows[i,:,:,:]**2)) for i in range(len(predicted_flows))]

print('Avg endpoint error',np.mean(EE))
