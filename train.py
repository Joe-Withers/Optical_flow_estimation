version = 'v002'

if version=='v001':
    import v001.datasets as ds
    from v001.UnSupFlowNet import UnSupFlowNet

if version=='v002':
    import v002.datasets as ds
    from v002.UnSupFlowNet import UnSupFlowNet

def load_dataset_filenames(im_path, dataset_name, n_data = 10, im_width = 128, im_height = 128):
    #load data
    if dataset_name=='flying_chairs':
        im1_filenames,im2_filenames = ds.load_synthetic_chairs_image_filenames(im_path, n_data=n_data)
        y_filenames = ds.load_synthetic_chairs_flows_filenames(im_path, n_data=n_data)
    else:
        im1_filenames,im2_filenames = ds.load_my_synthetic_image_filenames(im_path, n_data=n_data)
        y_filenames = ds.load_my_synthetic_flows_filenames(im_path, n_data=n_data)
    return im1_filenames, im2_filenames, y_filenames

#dataset info
# dataset_name = 'flying_chairs'
dataset_name = 'my_synthetic'
im_path = './data/obj_texture/train/'
n_data = 10
im_width = 128
im_height = 128
im1_filenames, im2_filenames, y_filenames = load_dataset_filenames(im_path, dataset_name, n_data = n_data, im_width = im_width, im_height = im_height)

#feedback paths
load_model_path = 'ignore'
save_model_path = 'ignore'
dump_path = './temp_analysis/'
#hyperparameters
hm_epochs = 1
epsilon = 0.005
lambda1 = 0.00000005
batch_size = 16
run_supervised=True
#train model
flow_estimator = UnSupFlowNet()
flow_estimator.train_network(im1_filenames, im2_filenames, im_width, im_height, y=y_filenames, dataset_name=dataset_name, t=1, hm_epochs = hm_epochs,
                            epsilon = epsilon, lambda1 = lambda1, batch_size = batch_size, run_supervised=run_supervised,
                            load_model_path = 'ignore', save_model_path = 'ignore', save_flow_im_path = dump_path, save_step = 1, show_step=1)
