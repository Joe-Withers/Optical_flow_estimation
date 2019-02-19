import tensorflow as tf
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import time
import tensorflow.contrib.slim as slim

def show_stuff(batch_x, batch_y, prd_flow, wrp_im):
    cv2.destroyAllWindows()
    pred_im = np.array(wrp_im[-1,:,:])
    orig_im = np.array(batch_x[-1,:,:,:])
    label_im = np.array(batch_y[-1,:,:,:])
    flow_im = visualise_flow(prd_flow[-1,:,:,:])
    cv2.imshow( "original im1", cv2.resize(orig_im[:,:,0],(300,300), interpolation = cv2.INTER_NEAREST));
    cv2.imshow( "original im2", cv2.resize(orig_im[:,:,1],(300,300), interpolation = cv2.INTER_NEAREST));
    cv2.imshow( "flow x direction image", cv2.resize(prd_flow[-1,:,:,0],(300,300), interpolation = cv2.INTER_NEAREST));
    cv2.imshow( "flow y direction image", cv2.resize(prd_flow[-1,:,:,1],(300,300), interpolation = cv2.INTER_NEAREST));
    cv2.imshow( "flow prediction", cv2.resize(flow_im,(300,300), interpolation = cv2.INTER_NEAREST));
    cv2.imshow( "prediction im2", cv2.resize(pred_im,(300,300), interpolation = cv2.INTER_NEAREST));
    cv2.imshow( "real im2", cv2.resize(label_im,(300,300), interpolation = cv2.INTER_NEAREST));
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_flows(flow, save_flow_im_path, start, gs):
    (n_data,_,_,_) = flow.shape
    for idx in range(0,n_data):
        cv2.imwrite(save_flow_im_path+str(start+idx)+'-global-step'+str(gs)+'.png',
            cv2.resize(visualise_flow(flow[idx,:,:,:]), (300,300), interpolation = cv2.INTER_NEAREST))

def visualise_flow(flow):
    [width, height, _] = flow.shape
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((width, height, 3))
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype('uint8'),cv2.COLOR_HSV2BGR)
    return bgr

def warp_image(im, f_pred, t):
    flow = tf.multiply(f_pred,t)
    im2 = tf.contrib.image.dense_image_warp(im,flow,name='dense_image_warp')
    return im2

def warping_error(im_to_warp, true_warped_im, flow, t):
    [_,w,h,_] = flow.shape
    if len(im_to_warp.shape) < 4:
        im_to_warp = tf.expand_dims(im_to_warp, axis=3)
    if len(true_warped_im.shape) < 4:
        true_warped_im = tf.expand_dims(true_warped_im, axis=3)
    warped_im = warp_image(tf.image.resize_images(im_to_warp,[w,h]), flow, t)
    return tf.losses.absolute_difference(tf.image.resize_images(true_warped_im,[w,h]), warped_im), warped_im

def smoothness_error(f_pred):
    U = tf.expand_dims(f_pred[:,:,:,0], axis=3)
    V = tf.expand_dims(f_pred[:,:,:,1], axis=3)

    gx = tf.expand_dims(tf.expand_dims([[-1.0, 1.0],[-1.0, 1.0]], axis=-1),axis=-1)
    gy = tf.expand_dims(tf.expand_dims([[-1.0, -1.0],[1.0, 1.0]], axis=-1),axis=-1)
    print(gx.shape)
    print('aaaabbba')

    Ux = tf.nn.conv2d(U, gx, [1,1,1,1], "SAME")
    Uy = tf.nn.conv2d(U, gy, [1,1,1,1], "SAME")
    Vx = tf.nn.conv2d(V, gx, [1,1,1,1], "SAME")
    Vy = tf.nn.conv2d(V, gy, [1,1,1,1], "SAME")

    Ug2 = tf.add(tf.pow(Ux, 2), tf.pow(Uy, 2))
    Vg2 = tf.add(tf.pow(Vx, 2), tf.pow(Vy, 2))

    return tf.reduce_sum(tf.add(Ug2, Vg2))

def FLowNetSimple(data):
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

    # 12 * 16 flow
    deconv5 = slim.conv2d_transpose(conv6_1, 512, [4, 4], 2, scope='deconv5')
    deconvflow6 = slim.conv2d_transpose(predict6, 2, [4, 4], 2, 'SAME', scope='deconvflow6')
    concat5 = tf.concat((conv5_1, deconv5, deconvflow6), axis=3, name='concat5')
    predict5 = slim.conv2d(concat5, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict5')
    # 24 * 32 flow
    deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], 2, 'SAME', scope='deconv4')
    deconvflow5 = slim.conv2d_transpose(predict5, 2, [4, 4], 2, 'SAME', scope='deconvflow5')
    concat4 = tf.concat((conv4_1, deconv4, deconvflow5), axis=3, name='concat4')
    predict4 = slim.conv2d(concat4, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict4')
    # 48 * 64 flow
    deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], 2, 'SAME', scope='deconv3')
    deconvflow4 = slim.conv2d_transpose(predict4, 2, [4, 4], 2, 'SAME', scope='deconvflow4')
    concat3 = tf.concat((conv3_1, deconv3, deconvflow4), axis=3, name='concat3')
    predict3 = slim.conv2d(concat3, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict3')
    # 96 * 128 flow
    deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], 2, 'SAME', scope='deconv2')
    deconvflow3 = slim.conv2d_transpose(predict3, 2, [4, 4], 2, 'SAME', scope='deconvflow3')
    concat2 = tf.concat((conv2, deconv2, deconvflow3), axis=3, name='concat2')
    predict2 = slim.conv2d(concat2, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict2')
    # 192 * 256 flow
    deconv1 = slim.conv2d_transpose(concat2, 64, [4, 4], 2, 'SAME', scope='deconv1')
    deconvflow2 = slim.conv2d_transpose(predict2, 2, [4, 4], 2, 'SAME', scope='deconvflow2')
    concat1 = tf.concat((conv1, deconv1, deconvflow2), axis=3, name='concat1')
    predict1 = slim.conv2d(concat1, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict1')

    return (predict1, predict3, predict2, predict4, predict5, predict6)


class SIFN():

    def __init__(self, load_model_path = 'ignore', save_model_path = 'ignore'):
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path

    def train_network(self, train_x, train_y, t = 1, hm_epochs = 100000, epsilon = 0.005, lambda1 = 0.00000005, batch_size = 16,
    save_flow_im_path = 'ignore', save_step = 100, show_step=100, bool_show_stuff = False):
        [_, width_of_image, height_of_image, n_x] = train_x.shape
        [_, width_of_image, height_of_image, n_y] = train_y.shape
        #placeholders
        x = tf.placeholder('float', [None, width_of_image, height_of_image, n_x])
        y = tf.placeholder('float', [None, width_of_image, height_of_image, n_y])
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #estimate flow
        (flow1, flow2, flow3, flow4, flow5, flow6) = FLowNetSimple(x)
        #calculate error
        w_err1, p_im1 = warping_error(x[:,:,:,0], y[:,:,:,0], flow1, t)
        w_err2, p_im2 = warping_error(x[:,:,:,0], y[:,:,:,0], flow2, t)
        w_err3, p_im3 = warping_error(x[:,:,:,0], y[:,:,:,0], flow3, t)
        w_err4, p_im4 = warping_error(x[:,:,:,0], y[:,:,:,0], flow4, t)
        w_err5, p_im5 = warping_error(x[:,:,:,0], y[:,:,:,0], flow5, t)
        w_err6, p_im6 = warping_error(x[:,:,:,0], y[:,:,:,0], flow6, t)
        p_flow = flow1
        p_image = p_im1
        weight = [1/2,      1/4,        1/8,        1/16,       1/32,       1/32]
        w_errs = [w_err1,   w_err2,     w_err3,     w_err4,     w_err5,     w_err6]
        w_err = tf.reduce_sum(tf.multiply(weight,w_errs))
        charbonnier_loss = tf.sqrt(tf.square(w_err) + tf.square(epsilon))
        #calculate smoothness error
        s_err1 = smoothness_error(flow1)
        s_err2 = smoothness_error(flow2)
        s_err3 = smoothness_error(flow3)
        s_err4 = smoothness_error(flow4)
        s_err5 = smoothness_error(flow5)
        s_err6 = smoothness_error(flow6)
        s_errs = [s_err1,   s_err2,     s_err3,     s_err4,     s_err5,     s_err6]
        s_err = tf.reduce_sum(tf.multiply(weight,s_errs))

        cost = tf.add(charbonnier_loss, lambda1*s_err)
        tf.summary.scalar("cost", cost)
        #optimise error
        optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)

        writer = tf.summary.FileWriter("/train/cost")
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
            saver = tf.train.Saver(max_to_keep=2)
            #load model from latest checkpoint
            if (self.load_model_path != 'ignore'):
                saver.restore(sess, tf.train.latest_checkpoint(self.load_model_path))
                print('model loaded from:', self.load_model_path)

            #training
            for epoch in range(hm_epochs):
                epoch_loss = 0
                i = 0
                start_t_epoch = time.time()
                while (i < len(train_x)) & (i < len(train_y)):
                    start = i
                    end = i+batch_size
                    start_t_batch = time.time()
                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])
                    _, c, prd_flow, wrp_im, g_s, summ = sess.run([optimizer, cost, p_flow, p_image, global_step, summaries],
                    feed_dict = {x: batch_x, y: batch_y})
                    end_t_batch = time.time()
                    writer.add_summary(summ, global_step=g_s)
                    print('time for batch:',end_t_batch - start_t_batch)
                    epoch_loss += c
                    i += batch_size

                    if(((epoch) % save_step) == 0) and (save_flow_im_path != 'ignore'):
                        save_flows(prd_flow, save_flow_im_path, start, g_s)

                end_t_epoch = time.time()
                print('Global Step:', g_s, 'Epoch:', epoch, '/', hm_epochs,
                'loss:', epoch_loss, 'time:', end_t_epoch - start_t_epoch)

                if(((epoch) % show_step) == 0) and bool_show_stuff:
                    show_stuff(batch_x, batch_y, prd_flow, wrp_im)

                #save model
                if (self.save_model_path != 'ignore'):
                    if(((epoch) % save_step) == 0):
                        saver.save(sess, self.save_model_path+'f_model', global_step=global_step)
                        print('model saved in:', self.save_model_path)

    def test_network(self, train_x, train_y, t = 1, hm_epochs = 100000, epsilon = 0.005, lambda1 = 0.00000005, batch_size = 16,
    save_flow_im_path = 'ignore', save_step = 100, show_step=100, bool_show_stuff = False):
        [_, width_of_image, height_of_image, n_x] = train_x.shape
        [_, width_of_image, height_of_image, n_y] = train_y.shape
        #placeholders
        x = tf.placeholder('float', [None, width_of_image, height_of_image, n_x])
        y = tf.placeholder('float', [None, width_of_image, height_of_image, n_y])
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #estimate flow
        (flow1, flow2, flow3, flow4, flow5, flow6) = FLowNetSimple(x)
        #calculate error
        w_err1, p_im1 = warping_error(x[:,:,:,0], y[:,:,:,0], flow1, t)
        w_err2, p_im2 = warping_error(x[:,:,:,0], y[:,:,:,0], flow2, t)
        w_err3, p_im3 = warping_error(x[:,:,:,0], y[:,:,:,0], flow3, t)
        w_err4, p_im4 = warping_error(x[:,:,:,0], y[:,:,:,0], flow4, t)
        w_err5, p_im5 = warping_error(x[:,:,:,0], y[:,:,:,0], flow5, t)
        w_err6, p_im6 = warping_error(x[:,:,:,0], y[:,:,:,0], flow6, t)
        p_flow = flow1
        p_image = p_im1
        weight = [1/2,      1/4,        1/8,        1/16,       1/32,       1/32]
        w_errs = [w_err1,   w_err2,     w_err3,     w_err4,     w_err5,     w_err6]
        w_err = tf.reduce_sum(tf.multiply(weight,w_errs))
        charbonnier_loss = tf.sqrt(tf.square(w_err) + tf.square(epsilon))
        #calculate smoothness error
        s_err1 = smoothness_error(flow1)
        s_err2 = smoothness_error(flow2)
        s_err3 = smoothness_error(flow3)
        s_err4 = smoothness_error(flow4)
        s_err5 = smoothness_error(flow5)
        s_err6 = smoothness_error(flow6)
        s_errs = [s_err1,   s_err2,     s_err3,     s_err4,     s_err5,     s_err6]
        s_err = tf.reduce_sum(tf.multiply(weight,s_errs))

        cost = tf.add(charbonnier_loss, lambda1*s_err)
        tf.summary.scalar("cost", cost)

        writer = tf.summary.FileWriter("/test/cost")
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
            saver = tf.train.Saver(max_to_keep=2)
            #load model from latest checkpoint
            if (self.load_model_path != 'ignore'):
                saver.restore(sess, tf.train.latest_checkpoint(self.load_model_path))
                print('model loaded from:', self.load_model_path)

            #training
            for epoch in range(hm_epochs):
                epoch_loss = 0
                i = 0
                start_t_epoch = time.time()
                while (i < len(train_x)) & (i < len(train_y)):
                    start = i
                    end = i+batch_size
                    start_t_batch = time.time()
                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])
                    c, prd_flow, wrp_im, g_s, summ = sess.run([cost, p_flow, p_image, global_step, summaries],
                    feed_dict = {x: batch_x, y: batch_y})
                    end_t_batch = time.time()
                    writer.add_summary(summ, global_step=g_s)
                    print('time for batch:',end_t_batch - start_t_batch)
                    epoch_loss += c
                    i += batch_size

                    if(((epoch) % save_step) == 0) and (save_flow_im_path != 'ignore'):
                        save_flows(prd_flow, save_flow_im_path, start, g_s)

                end_t_epoch = time.time()
                print('Global Step:', g_s, 'Epoch:', epoch, '/', hm_epochs,
                'loss:', epoch_loss, 'time:', end_t_epoch - start_t_epoch)

                if(((epoch) % show_step) == 0) and bool_show_stuff:
                    show_stuff(batch_x, batch_y, prd_flow, wrp_im)
