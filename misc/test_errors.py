import sys
sys.path.append('./')
import numpy as np
import tensorflow as tf
version = 'v002'

if version=='v001':
    import v001.datasets as ds
    from v001.UnSupFlowNet import UnSupFlowNet

if version=='v002':
    import v002.datasets as ds
    from v002.UnSupFlowNet import UnSupFlowNet

def angular_error(flow_gt,flow):
    [x,y,_] = flow.shape
    f = np.concatenate((flow, np.ones((x,y,1))),axis=2)
    f_gt = np.concatenate((flow_gt, np.ones((x,y,1))),axis=2)
    #arccos doesnt seem to handle edge case of all 1.0
    ae = np.arccos( (np.sum(f*f_gt, axis=2)) / ( np.sqrt(np.sum(f**2, axis=2)) * np.sqrt(np.sum(f_gt**2, axis=2)) ) )
    return ae

def load_dataset_filenames(im_path, dataset_name, n_data = 10):
    #load data
    if dataset_name=='flying_chairs':
        im1_filenames,im2_filenames = ds.load_synthetic_chairs_image_filenames(im_path, n_data=n_data)
        y_filenames = ds.load_synthetic_chairs_flows_filenames(im_path, n_data=n_data)
    else:
        im1_filenames,im2_filenames = ds.load_my_synthetic_image_filenames(im_path, n_data=n_data)
        y_filenames = ds.load_my_synthetic_flows_filenames(im_path, n_data=n_data)
    return im1_filenames, im2_filenames, y_filenames

models = ['sup_5e-8','sup_5e-10','sup_0','unsup_5e-6','unsup_5e-8','unsup_5e-10','unsup_0']
models = ['unsup_5e-6']
dataset_name = 'flying_chairs'
model_dataset_name = 'flying_chairs'

n_data = 100
im_path = './data/FlyingChairs2/train/'

im1_filenames, im2_filenames, y_filenames = load_dataset_filenames(im_path, dataset_name, n_data = n_data)

#load data
# X = ds.load_my_synthetic_images(im_path, n_data=n_data)
# GT_flows = ds.load_my_synthetic_flows(im_path, n_data=n_data)
# X = ds.load_synthetic_chairs_images(im_path, n_data=n_data)
# GT_flows = ds.load_synthetic_chairs_flows(im_path, n_data=n_data)
AAE = dict()
AEE = dict()
for model in models:
    try:
        print('aaaa',model)
        #feedback paths
        load_model_path = './'+version+'/training/kaggle/'+model_dataset_name+'/'+model+'/'
        dump_path = 'ignore'
        #feedback paras
        save_step = 30
        show_step = 10
        bool_show_stuff = False
        #training info
        batch_size = 32
        #train model
        flow_estimator = UnSupFlowNet()
        tf.reset_default_graph()
        # predicted_flows = flow_estimator.run_network_with_data(X, load_model_path, batch_size=batch_size, save_flow_im_path=dump_path)
        ee, ae = flow_estimator.run_network(im1_filenames, im2_filenames, load_model_path, y=y_filenames ,dataset_name=dataset_name, batch_size = batch_size, save_flow_im_path = dump_path)
        print(predicted_flows.shape)
        [a1,b1,c1,d1] = GT_flows.shape
        [a2,b2,c2,d2] = predicted_flows.shape
        assert a1==a2
        assert b1==b2
        assert c1==c2
        assert d1==d2
        AE = [np.mean(angular_error(GT_flows[i,:,:,:],predicted_flows[i,:,:,:])) for i in range(len(predicted_flows))]
        AAE[model] = np.mean(AE)
        print(model,': Avg Angular error',np.mean(AE))

        EE = [np.mean(np.sqrt(GT_flows[i,:,:,:]**2+predicted_flows[i,:,:,:]**2)) for i in range(len(predicted_flows))]
        AEE[model] = np.mean(EE)
        print(model,': Avg Endpoint error',np.mean(EE))
    except:
        print('error')

print('************Result summary*************')
for model in models:
    try:
        print(model,': Avg Endpoint error',AEE[model])
        print(model,': Avg Angular error',AAE[model])
    except:
        print('error')
