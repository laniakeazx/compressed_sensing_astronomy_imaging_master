import numpy as np
from matplotlib.pylab import rcParams
from ipywidgets.widgets import *
from sparsity_tool import plot
from sparsity_tool.functions import l1_norm, sigma_mad,nfft,infft,mask_op,upsample,soft_thresh,cost_func,forwardBackward,grad,solve,nmse

rcParams['figure.figsize'] = (14.0, 8.0)
image_shape = (256, 256)
mask2d = np.load('data/cs_mask.npy')
print("mask2d shape",np.shape(mask2d))
y2d = np.load('data/cs_obs_data.npy')
print("y2d shape",np.shape(y2d))
plot.display(mask2d, 'Mask', shape=image_shape, cmap='jet')
plot.display(y2d, r'$y$', shape=(32768, 1), cmap='jet')
print('The percentage of samples used is:', mask2d[mask2d == 1].size / float(mask2d.size) * 100)
print("y2d shape",np.shape(y2d))
print("mask2d shape",np.shape(mask2d))
plot.display(upsample(y2d, mask2d), r'$y$', shape=image_shape, cmap='jet')
# Recover the sparse singal alpha
alpha_rec2d, cost2d = forwardBackward(y2d, first_guess=np.ones(mask2d.size), mask=mask2d, grad=grad, lambda_val=0.09,
                                      n_iter=5000, return_cost=True)
x_rec2d = nfft(alpha_rec2d)
# Display the recovered signal x
plot.display(x_rec2d, title=r'$\hat{x}$', shape=image_shape, cmap='jet')
# Display the cost function
plot.cost_plot(cost2d)
print('NMSE =', nmse(np.load('data/cs_true_data.npy'), x_rec2d))



