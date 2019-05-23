version = 'v002'

if version=='v001':
    import v001.datasets as ds
    from v001.UnSupFlowNet import UnSupFlowNet

if version=='v002':
    import v002.datasets as ds
    from v002.UnSupFlowNet import UnSupFlowNet

def load_dataset_filenames(im_path, dataset_name, n_data = 10):
    #load data
    if dataset_name=='flying_chairs':
        im1_filenames,im2_filenames = ds.load_synthetic_chairs_image_filenames(im_path, n_data=n_data)
    else:
        im1_filenames,im2_filenames = ds.load_my_synthetic_image_filenames(im_path, n_data=n_data)
    return im1_filenames, im2_filenames

dataset_name = 'flying_chairs'
im_path = './data/FlyingChairs2/train/'
n_data = 50
im1_filenames, im2_filenames = load_dataset_filenames(im_path, dataset_name, n_data = n_data)

#feedback paths
load_model_path = './'+version+'/training/kaggle/flying_chairs/unsup_5e-6(occ_unsup_5e-6)/'
dump_path = './temp_analysis/'
#train model
flow_estimator = UnSupFlowNet()
flow_estimator.run_network(im1_filenames, im2_filenames, load_model_path, dataset_name=dataset_name, batch_size = 4, save_flow_im_path = dump_path)
