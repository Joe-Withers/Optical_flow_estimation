import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import numpy as np

SAVE_FIG = True
dir = './v001/training/kaggle/'
# filename_dict = {'obj_texture_sup':['_1','_2'],
#             'obj_texture_unsup_5e-8':['_1','_2'],
#             'obj_texture_unsup_5e-10':['_1'],
#             'obj_texture_unsup_0':['_1']}
# filename_dict = {'no_texture_sup':['_1'],
#             'no_texture_unsup_5e-8':['_1'],
#             'no_texture_unsup_5e-10':['_1'],
#             'no_texture_unsup_0':['_1']}
# filename_dict = {'flying_chairs_sup_5e-8_32':[''],
#             'flying_chairs_unsup_5e-8_32':[''],
#             'flying_chairs_unsup_5e-10_32':[''],
#             'flying_chairs_unsup_0_32':['']}
# title_postfix = '\non the flying chairs dataset'
filename_dict = {'affine_sup_5e-8_16':[''],
            'affine_unsup_5e-8_16':[''],
            'affine_unsup_5e-10_16':[''],
            'affine_unsup_0_16':['']}
title_postfix = '\non the affine transformation dataset'
# filename_dict = {'obj_texture_sup':[''],
#             'obj_texture_unsup_5e-8_16':[''],
#             'obj_texture_unsup_5e-10_16':[''],
#             'obj_texture_unsup_0_16':['']}
# title_postfix = '\non the synthetic object texture dataset'
# filename_dict = {'no_texture_sup':[''],
#             'no_texture_unsup_5e-8_16':[''],
#             'no_texture_unsup_5e-10_16':[''],
#             'no_texture_unsup_0_16':['']}
# title_postfix = '\non the synthetic no texture dataset'

legend = ['Supervised: λ = 5e-8','Unsupervised: λ = 5e-8','Unsupervised: λ = 5e-10','Unsupervised: λ = 0']
loss_dict = dict()
ploss_dict = dict()
floss_dict = dict()
for k in filename_dict.keys():
    loss = np.array([])
    ploss = np.array([])
    floss = np.array([])
    for post_fix in filename_dict[k]:
        n_loss = pickle.load(open(dir+k+'/loss'+post_fix+'.pickle', 'rb'))
        n_ploss = pickle.load(open(dir+k+'/ploss'+post_fix+'.pickle', 'rb'))
        n_floss = pickle.load(open(dir+k+'/floss'+post_fix+'.pickle', 'rb'))
        loss = np.concatenate((loss,n_loss))
        ploss = np.concatenate((ploss,n_ploss))
        floss = np.concatenate((floss,n_floss))
    loss_dict[k] = loss
    ploss_dict[k] = ploss
    floss_dict[k] = floss

def average_batches(x, batches_per_epoch):
    x = np.array(x)
    x = x.reshape(-1, batches_per_epoch)
    x_new = [np.mean(xi) for xi in x]
    x_new = np.array(x_new)
    return x_new

def batch_to_epoch(l_dict, batches_per_epoch):
    for k in l_dict:
        l_dict[k] = average_batches(l_dict[k], batches_per_epoch)
    return l_dict

# bpe = 5558
# loss_dict = batch_to_epoch(loss_dict, bpe)
# ploss_dict = batch_to_epoch(ploss_dict, bpe)
# floss_dict = batch_to_epoch(floss_dict, bpe)

#plot loss
for k in loss_dict.keys():
    plt.plot(gaussian_filter(loss_dict[k], sigma=10))
plt.title('Blurred loss learning curve'+title_postfix)
plt.ylabel('loss')
plt.xlabel('Steps (batch size = 32)')
plt.legend(legend)
if SAVE_FIG:
    plt.savefig(dir+'loss.png')
plt.show()

#plot photometric loss
for k in ploss_dict.keys():
    plt.plot(gaussian_filter(ploss_dict[k], sigma=10))
plt.title('Blurred photometric loss learning curve'+title_postfix)
plt.ylabel('Photometric loss')
plt.xlabel('Steps (batch size = 32)')
plt.legend(legend)
if SAVE_FIG:
    plt.savefig(dir+'ploss.png')
plt.show()

#plot flow loss
for k in floss_dict.keys():
    plt.plot(gaussian_filter(floss_dict[k], sigma=10))
plt.title('Blurred flow loss learning curve'+title_postfix)
plt.ylabel('Flow loss')
plt.xlabel('Steps (batch size = 32)')
plt.legend(legend)
if SAVE_FIG:
    plt.savefig(dir+'floss.png')
plt.show()
