import tensorflow as tf
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import time
import tensorflow.contrib.slim as slim
import pickle

########################## helper functions ################################
def show_stuff(batch_x, batch_y, prd_flow, wrp_im):
    cv2.destroyAllWindows()
    for i in range(0,3):
        pred_im = np.array(wrp_im[i,:,:])
        orig_im = np.array(batch_x[i,:,:,:])
        label_im = visualise_flow(batch_y[i,:,:,:])
        flow_im = visualise_flow(prd_flow[i,:,:,:])
        cv2.imshow( "original im1", cv2.resize(orig_im[:,:,0],(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.imshow( "original im2", cv2.resize(orig_im[:,:,1],(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.imshow( "flow x direction image", cv2.resize(flow_im[:,:,0],(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.imshow( "flow y direction image", cv2.resize(flow_im[:,:,1],(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.imshow( "flow prediction", cv2.resize(flow_im,(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.imshow( "prediction im2", cv2.resize(pred_im,(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.imshow( "real flow", cv2.resize(label_im,(300,300), interpolation = cv2.INTER_NEAREST));
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualise_flow(flow):
    [width, height, _] = flow.shape
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((width, height, 3))
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype('uint8'),cv2.COLOR_HSV2BGR)
    return bgr

def read_flow_chairs(im_path):
    f = open(im_path, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    flow_t = tf.reverse(flow, [-1])
    return flow_t

def read_flow_npy(im_path):
    f_flow = im_path
    flow = np.load(f_flow).astype(np.float32)
    return 0.5*flow

def set_shape_fn(input, shape):
    input.set_shape(shape)
    return input

########################## Warping function ################################
def warp_image(im, f_pred):
    im_warped = tf.contrib.image.dense_image_warp(im,f_pred,name='dense_image_warp')
    im_warped.set_shape(im.shape)
    return im_warped

########################## Error functions ################################
def flow_error(gt_flow, flow):
    [_,w,h,_] = flow.shape
    resized_gt_flow = tf.image.resize_images(gt_flow,[w,h])
    return tf.reduce_mean(tf.abs(resized_gt_flow - flow))

def smoothness_error(f_pred):
    U = tf.expand_dims(f_pred[:,:,:,0], axis=3)
    V = tf.expand_dims(f_pred[:,:,:,1], axis=3)

    gx = tf.expand_dims(tf.expand_dims([[-1.0, 1.0],[-1.0, 1.0]], axis=-1),axis=-1)
    gy = tf.expand_dims(tf.expand_dims([[-1.0, -1.0],[1.0, 1.0]], axis=-1),axis=-1)

    Ux = tf.nn.conv2d(U, gx, [1,1,1,1], "SAME")
    Uy = tf.nn.conv2d(U, gy, [1,1,1,1], "SAME")
    Vx = tf.nn.conv2d(V, gx, [1,1,1,1], "SAME")
    Vy = tf.nn.conv2d(V, gy, [1,1,1,1], "SAME")

    Ug2 = tf.add(tf.pow(Ux, 2), tf.pow(Uy, 2))
    Vg2 = tf.add(tf.pow(Vx, 2), tf.pow(Vy, 2))

    return tf.reduce_mean(tf.add(Ug2, Vg2))

def warping_error(im_to_warp, true_warped_im, flow):
    if len(im_to_warp.shape) < 4:
        im_to_warp = tf.expand_dims(im_to_warp, axis=3)
    if len(true_warped_im.shape) < 4:
        true_warped_im = tf.expand_dims(true_warped_im, axis=3)
    [_,im_w,im_h,_] = im_to_warp.shape
    [_,f_w,f_h,_] = flow.shape
    im_to_warp = tf.image.resize_images(im_to_warp,[f_w,f_h])
    true_warped_im = tf.image.resize_images(true_warped_im,[f_w,f_h])
    warped_im = warp_image(im_to_warp, flow)
    return tf.reduce_mean(tf.abs(true_warped_im - warped_im)), warped_im

########################## FlowNetS network ################################
def FLowNetSimple(data):
    # link for code used in this function: https://github.com/linjian93/tf-flownet/blob/master/train_flownet_simple.py
    concat1 = data
    conv1 = slim.conv2d(concat1, 64, [7, 7], 2, scope='conv1')
    conv2 = slim.conv2d(conv1, 128, [5, 5], 2, scope='conv2')
    conv3 = slim.conv2d(conv2, 256, [5, 5], 2, scope='conv3')
    conv3_1 = slim.conv2d(conv3, 256, [3, 3], 1, scope='conv3_1')
    conv4 = slim.conv2d(conv3_1, 512, [3, 3], 2, scope='conv4')
    conv4_1 = slim.conv2d(conv4, 512, [3, 3], 1, scope='conv4_1')
    conv5 = slim.conv2d(conv4_1, 512, [3, 3], 2, scope='conv5')
    conv5_1 = slim.conv2d(conv5, 512, [3, 3], 1, scope='conv5_1')
    conv6 = slim.conv2d(conv5_1, 1024, [3, 3], 2, scope='conv6')
    conv6_1 = slim.conv2d(conv6, 1024, [3, 3], 1, scope='conv6_1')
    predict6 = slim.conv2d(conv6_1, 2, [3, 3], 1, activation_fn=None, scope='pred6')

    deconv5 = slim.conv2d_transpose(conv6_1, 512, [4, 4], 2, scope='deconv5')
    deconvflow6 = slim.conv2d_transpose(predict6, 2, [4, 4], 2, 'SAME', scope='deconvflow6')
    concat5 = tf.concat((conv5_1, deconv5, deconvflow6), axis=3, name='concat5')
    predict5 = slim.conv2d(concat5, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict5')

    deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], 2, 'SAME', scope='deconv4')
    deconvflow5 = slim.conv2d_transpose(predict5, 2, [4, 4], 2, 'SAME', scope='deconvflow5')
    concat4 = tf.concat((conv4_1, deconv4, deconvflow5), axis=3, name='concat4')
    predict4 = slim.conv2d(concat4, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict4')

    deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], 2, 'SAME', scope='deconv3')
    deconvflow4 = slim.conv2d_transpose(predict4, 2, [4, 4], 2, 'SAME', scope='deconvflow4')
    concat3 = tf.concat((conv3_1, deconv3, deconvflow4), axis=3, name='concat3')
    predict3 = slim.conv2d(concat3, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict3')

    deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], 2, 'SAME', scope='deconv2')
    deconvflow3 = slim.conv2d_transpose(predict3, 2, [4, 4], 2, 'SAME', scope='deconvflow3')
    concat2 = tf.concat((conv2, deconv2, deconvflow3), axis=3, name='concat2')
    predict2 = slim.conv2d(concat2, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict2')

    deconv1 = slim.conv2d_transpose(concat2, 64, [4, 4], 2, 'SAME', scope='deconv1')
    deconvflow2 = slim.conv2d_transpose(predict2, 2, [4, 4], 2, 'SAME', scope='deconvflow2')
    concat1 = tf.concat((conv1, deconv1, deconvflow2), axis=3, name='concat1')
    predict1 = slim.conv2d(concat1, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict1')
    return (predict1, predict3, predict2, predict4, predict5, predict6)

########################## main class ################################
class UnSupFlowNet():

    def __init__(self):
        pass

    def save_flows(self, flow, save_flow_im_path, first_batch, gs):
        (n_data,_,_,_) = flow.shape
        if first_batch:
            self.start_idx = 0
        start = self.start_idx
        if start < 50:
            for idx in range(0,n_data):
                cv2.imwrite(save_flow_im_path+str(start+idx)+'-global-step'+str(gs)+'.png',
                    cv2.resize(visualise_flow(flow[idx,:,:,:]), (300,300), interpolation = cv2.INTER_NEAREST))
        self.start_idx += n_data

    def _dataset_pipeline(self, im1_filenames, im2_filenames, y, dataset_name, batch_size, w, h):
        # Make a Dataset of file names including all the PNG images files in
        # the relative image directory.
        filename_dataset_im1 = tf.data.Dataset.from_tensor_slices(im1_filenames)
        filename_dataset_im2 = tf.data.Dataset.from_tensor_slices(im2_filenames)
        # Make a Dataset of image tensors by reading and decoding the files.
        image_dataset_im1 = filename_dataset_im1.map(lambda x: tf.image.decode_png(tf.read_file(x), channels=1) / 255 )
        image_dataset_im1 = image_dataset_im1.map(lambda x: set_shape_fn(x,(w,h,1)))

        image_dataset_im2 = filename_dataset_im2.map(lambda x: tf.image.decode_png(tf.read_file(x), channels=1) / 255 )
        image_dataset_im2 = image_dataset_im2.map(lambda x: set_shape_fn(x,(w,h,1)))

        if y is not None:
            dataset_y = tf.data.Dataset.from_tensor_slices(y)
            if dataset_name=='flying_chairs':
                dataset_y = dataset_y.map(lambda filename: tf.py_func(read_flow_chairs, [filename], [tf.float32]))
            else:
                dataset_y = dataset_y.map(lambda filename: tf.py_func(read_flow_npy, [filename], [tf.float32]))
            dataset_y = dataset_y.map(lambda flow: set_shape_fn(flow,(w,h,2)))
            image_dataset = tf.data.Dataset.zip((image_dataset_im1, image_dataset_im2, dataset_y))
        else:
            image_dataset = tf.data.Dataset.zip((image_dataset_im1, image_dataset_im2))
        #batch data points
        image_dataset = image_dataset.batch(batch_size)

        iterator = image_dataset.make_initializable_iterator()
        return iterator

    def train_network(self, im1_filenames, im2_filenames, y=None, dataset_name = 'my_synthetic', hm_epochs = 1,
    epsilon = 0.005, lambda1 = 0.00000005, batch_size = 16, run_supervised=True,
    load_model_path = 'ignore', save_model_path = 'ignore', save_flow_im_path = 'ignore', save_step = 100, show_step=-1):
        """ trains a model """
        if dataset_name=='flying_chairs':
            width = 384
            height = 512
        else:
            width = 128
            height = 128
        #boolean for whether or not to show stuff whie training - mainly used for debugging or to check it's working as expected
        bool_show_stuff = show_step > 0
        if run_supervised and (y is None):
            print('No flow labels, running UNsupervised')
            run_supervised = False
        #placeholders-
        iterator = self._dataset_pipeline(im1_filenames, im2_filenames, y, dataset_name, batch_size, width, height)
        if y is not None:
            im1, im2, gt_flow = iterator.get_next()
        else:
            im1, im2 = iterator.get_next()

        x = tf.concat((im1,im2), axis=3)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #estimate flow
        (flow1, flow2, flow3, flow4, flow5, flow6) = FLowNetSimple(x)
        weight = [1/2,      1/4,        1/8,        1/16,       1/32,       1/32]
        #calculate smoothness error
        s_err1 = smoothness_error(flow1)
        s_err2 = smoothness_error(flow2)
        s_err3 = smoothness_error(flow3)
        s_err4 = smoothness_error(flow4)
        s_err5 = smoothness_error(flow5)
        s_err6 = smoothness_error(flow6)
        s_errs = [s_err1,   s_err2,     s_err3,     s_err4,     s_err5,     s_err6]
        s_err = tf.reduce_sum(tf.multiply(weight,s_errs))
        #calculate unsupervised error
        w_err1, p_im1 = warping_error(x[:,:,:,1], x[:,:,:,0], -flow1 / 2**1)
        w_err2, p_im2 = warping_error(x[:,:,:,1], x[:,:,:,0], -flow2 / 2**2)
        w_err3, p_im3 = warping_error(x[:,:,:,1], x[:,:,:,0], -flow3 / 2**3)
        w_err4, p_im4 = warping_error(x[:,:,:,1], x[:,:,:,0], -flow4 / 2**4)
        w_err5, p_im5 = warping_error(x[:,:,:,1], x[:,:,:,0], -flow5 / 2**5)
        w_err6, p_im6 = warping_error(x[:,:,:,1], x[:,:,:,0], -flow6 / 2**6)
        w_errs = [w_err1,   w_err2,     w_err3,     w_err4,     w_err5,     w_err6]
        w_err = tf.reduce_sum(tf.multiply(weight,w_errs))
        if y is not None:
        #calculate supervised error
            f_err1 = flow_error(gt_flow, flow1)
            f_err2 = flow_error(gt_flow, flow2)
            f_err3 = flow_error(gt_flow, flow3)
            f_err4 = flow_error(gt_flow, flow4)
            f_err5 = flow_error(gt_flow, flow5)
            f_err6 = flow_error(gt_flow, flow6)
            f_errs = [f_err1,   f_err2,     f_err3,     f_err4,     f_err5,     f_err6]
            f_err = tf.reduce_sum(tf.multiply(weight,f_errs))

        if run_supervised:
            charbonnier_loss = tf.sqrt(tf.square(f_err) + tf.square(epsilon))
            cost = tf.add(charbonnier_loss, lambda1*s_err)
        else:
            charbonnier_loss = tf.sqrt(tf.square(w_err) + tf.square(epsilon))
            cost = tf.add(charbonnier_loss, lambda1*s_err)

        tf.summary.scalar("cost", cost)
        #optimise error
        optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)

        #stuff to keep track of
        p_flow = tf.image.resize_images(flow1, [width, height])
        p_image = tf.image.resize_images(p_im1, [width, height])
        avg_abs_photometric_loss = tf.losses.absolute_difference(x[:,:,:,0], p_image[:,:,:,0])
        if y is not None:
            avg_abs_flow_loss = tf.losses.absolute_difference(gt_flow, p_flow)
        writer = tf.summary.FileWriter("./train/cost")
        summaries = tf.summary.merge_all()

        #run on GPU
        config = tf.ConfigProto(
            device_count = {'GPU': 1}
        )
        with tf.Session(config=config) as sess:
            #initialise the model randomly
            print('initializing model')
            sess.run(tf.global_variables_initializer())

            # saver object to save the variables
            saver = tf.train.Saver(max_to_keep=1)
            #load model from latest checkpoint
            if (load_model_path != 'ignore'):
                saver.restore(sess, tf.train.latest_checkpoint(load_model_path))
                print('model loaded from:', load_model_path)

            all_loss = []
            all_ploss = []
            all_floss = []
            #training
            for epoch in range(hm_epochs):
                epoch_loss = 0
                epoch_ploss = 0
                epoch_floss = 0
                first_batch = True
                start_t_epoch = time.time()
                sess.run(iterator.initializer)
                while True:
                    try:
                        start_t_batch = time.time()
                        if y is not None:
                            ploss, floss, batch_x, batch_y, _, c, prd_flow, wrp_im, g_s, summ = sess.run([avg_abs_photometric_loss, avg_abs_flow_loss, x, gt_flow, optimizer, cost, p_flow, p_image, global_step, summaries])
                        else:
                            ploss, batch_x, _, c, prd_flow, wrp_im, g_s, summ = sess.run([avg_abs_photometric_loss, x, optimizer, cost, p_flow, p_image, global_step, summaries])
                            floss = 0
                        end_t_batch = time.time()
                        writer.add_summary(summ, global_step=g_s)
                        print('time for batch:',end_t_batch - start_t_batch)
                        epoch_loss += c
                        epoch_ploss += ploss
                        epoch_floss += floss
                        all_loss.append(c)
                        all_ploss.append(ploss)
                        all_floss.append(floss)
                        if(((epoch) % save_step) == 0) and (save_flow_im_path != 'ignore'):
                            self.save_flows(prd_flow, save_flow_im_path, first_batch, g_s)

                        if first_batch:
                            first_batch = False
                    except tf.errors.OutOfRangeError:
                        print('End of Epoch')
                        break

                end_t_epoch = time.time()
                print('Global Step:', g_s, 'Epoch:', epoch, '/', hm_epochs,
                'loss:', epoch_loss, epoch_ploss, epoch_floss, 'time:', end_t_epoch - start_t_epoch)
                if(((epoch) % show_step) == 0) and bool_show_stuff:
                    show_stuff(batch_x, batch_y, prd_flow, wrp_im)

                #save model
                if (save_model_path != 'ignore'):
                    pickle.dump(all_loss, open(save_model_path+'loss.pickle','wb'))
                    pickle.dump(all_ploss, open(save_model_path+'ploss.pickle','wb'))
                    pickle.dump(all_floss, open(save_model_path+'floss.pickle','wb'))
                    if(((epoch) % save_step) == 0):
                        saver.save(sess, save_model_path+'f_model', global_step=global_step)
                        print('model saved in:', save_model_path)

    def run_network(self, im1_filenames, im2_filenames, load_model_path, y=None, dataset_name='my_synthetic', batch_size = 16, save_flow_im_path = 'ignore'):
        """ runs a model """
        if dataset_name=='flying_chairs':
            width = 384
            height = 512
        else:
            width = 128
            height = 128
        #placeholders
        iterator = self._dataset_pipeline(im1_filenames, im2_filenames, y, dataset_name, batch_size, width, height)
        if y is not None:
            im1, im2, gt_flow = iterator.get_next()
        else:
            im1, im2 = iterator.get_next()
        x = tf.concat((im1,im2), axis=3)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #estimate flow
        (flow1, flow2, flow3, flow4, flow5, flow6) = FLowNetSimple(x)
        p_flow = tf.image.resize_images(flow1, [width, height])

        config = tf.ConfigProto(
            device_count = {'GPU': 1}
        )
        with tf.Session(config=config) as sess:
            #initialise the model randomly
            print('initializing model')
            sess.run(tf.global_variables_initializer())

            # saver object to save the variables
            saver = tf.train.Saver()
            #load model from latest checkpoint
            saver.restore(sess, tf.train.latest_checkpoint(load_model_path))
            print('model loaded from:', load_model_path)

            first_batch = True
            start_t_all = time.time()
            sess.run(iterator.initializer)
            #running
            flows = []
            ee=[]
            ae=[]
            while True:
                try:
                    start_t_batch = time.time()
                    prd_flow, g_s, grdt_flow = sess.run([p_flow, global_step, gt_flow])
                    end_t_batch = time.time()

                    if(save_flow_im_path != 'ignore'):
                        self.save_flows(prd_flow, save_flow_im_path, first_batch, g_s)


                    if first_batch:
                        # flows = prd_flow
                        ee = [np.mean(np.sqrt(grdt_flow[i,:,:,:]**2+prd_flow[i,:,:,:]**2)) for i in range(len(prd_flow))]
                        ae = [np.mean(angular_error(grdt_flow[i,:,:,:],prd_flow[i,:,:,:])) for i in range(len(prd_flow))]
                    else:
                        ee = np.concatenate((ee, [np.mean(np.sqrt(grdt_flow[i,:,:,:]**2+prd_flow[i,:,:,:]**2)) for i in range(len(prd_flow))]), axis=0)
                        ae = np.concatenate((ae, [np.mean(angular_error(grdt_flow[i,:,:,:],prd_flow[i,:,:,:])) for i in range(len(prd_flow))]), axis=0)
                        # flows = np.concatenate((flows, prd_flow), axis=0)
                    print('ee.shape',ee.shape)
                    print('ae.shape',ae.shape)
                    first_batch = False
                except tf.errors.OutOfRangeError:
                    print('End of Epoch')
                    break

            end_t_all = time.time()
            print('Time:', end_t_all - start_t_all, '\nAvg time per image pair:', (end_t_all - start_t_all)/len(flows))

            return ee,ae

    def run_network_with_data(self, test_x, load_model_path, batch_size = 32, save_flow_im_path = 'ignore'):
        """ runs a model - pass in actual data instead of paths to data """
        [n_ims, width_of_image, height_of_image, n_x] = test_x.shape
        #placeholders
        x = tf.placeholder('float', [None, width_of_image, height_of_image, n_x])
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #estimate flow
        (flow1, flow2, flow3, flow4, flow5, flow6) = FLowNetSimple(x)
        p_flow = tf.image.resize_images(flow1, [width_of_image, height_of_image])

        config = tf.ConfigProto(
            device_count = {'GPU': 1}
        )
        with tf.Session(config=config) as sess:
            #initialise the model randomly
            print('initializing model')
            sess.run(tf.global_variables_initializer())

            # saver object to save the variables
            saver = tf.train.Saver()
            #load model from latest checkpoint
            saver.restore(sess, tf.train.latest_checkpoint(load_model_path))
            print('model loaded from:', load_model_path)

            start_t_all = time.time()
            #running
            i = 0
            while i < len(test_x):
                start = i
                end = i+batch_size
                start_t_batch = time.time()
                batch_x = np.array(test_x[start:end])
                prd_flow, g_s = sess.run([p_flow, global_step],
                feed_dict = {x: batch_x})
                end_t_batch = time.time()
                i += batch_size

                if start == 0:
                    flows = prd_flow
                else:
                    flows = np.concatenate((flows, prd_flow), axis=0)

            end_t_all = time.time()
            if(save_flow_im_path != 'ignore'):
                self.save_flows(flows, save_flow_im_path, True, g_s)
            print('Time:', end_t_all - start_t_all, '\nAvg time per image pair:', (end_t_all - start_t_all)/n_ims)

            return flows
