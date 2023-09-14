import matplotlib.pyplot as plt

from astropy.visualization import astropy_mpl_style
import numpy
import matplotlib.pyplot as plt
from modopt.math.convolve import convolve
from modopt.signal.noise import add_noise
from PIL import Image
import numpy as np
from math import log
plt.style.use(astropy_mpl_style)
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from math import log
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker


image_file0 = fits.open('hogbomclean_25000ni_256256_natural_9_13image.fits')#m51_256_15000_clark_64arc_3sca3image
image_file0 = image_file0[0].data[0][0]
# image_file0=np.delete(image_file0,1,axis=0)
# image_file0 = 1*np.abs(image_file0-np.min(image_file0))/np.abs(np.max(image_file0)-np.min(image_file0))
print(image_file0.shape)

image_file1 = fits.open('hogbomclean_25000ni_256256_natural_9_13model.fits')#m51_256_15000_clark_64arc_3sca3model
image_file1 = image_file1[0].data[0][0]
print(image_file1.shape)
# image_file1 = 1*np.abs(image_file1-np.min(image_file1))/np.abs(np.max(image_file1)-np.min(image_file1))

image_file2 = fits.open('hogbomclean_25000ni_256256_natural_9_13residual.fits')#m51_256_15000_clark_64arc_3sca3residual
image_file2 = image_file2[0].data[0][0]
print(image_file2.shape)
# image_file2 = 1*np.abs(image_file2-np.min(image_file2))/np.abs(np.max(image_file2)-np.min(image_file2))

# RMSE1 = np.linalg.norm(image_file2,ord=2)/256
# maxrestore1=np.max(image_file0)
# print("max restored",maxrestore1)
# print(RMSE1,maxrestore1/RMSE1)

image_file3 = fits.open('MSclean_20000ni_256256_thre60mjy_natural2image.fits')#m51_256_15000_multiscale_64arc_3sca3image
image_file3 = image_file3[0].data[0][0]
print(image_file3.shape)
# image_file3 = 1*np.abs(image_file3-np.min(image_file3))/np.abs(np.max(image_file3)-np.min(image_file3))

image_file4 = fits.open('MSclean_20000ni_256256_thre60mjy_natural2model.fits')#m51_256_15000_multiscale_64arc_3sca3model
image_file4 = image_file4[0].data[0][0]
print(image_file4.shape)

image_file5 = fits.open('MSclean_20000ni_256256_thre60mjy_natural2residual.fits')#m51_256_15000_multiscale_64arc_3sca3residual
image_file5 = image_file5[0].data[0][0]
print(image_file5.shape)


# RMSE2 = np.linalg.norm(image_file5,ord=2)/256
# maxrestore2=np.max(image_file3)
# print(RMSE2,maxrestore2/RMSE2)
# print("max restored",maxrestore2)

image_file6 = fits.open('pf_iuwt_recove.fits')#condat_deconvolved_bar_extent
image_file6 = image_file6[0].data
print(image_file6.shape)

image_file7 = fits.open('MSclean_20000ni_256256model.fits')#condat_trueskymodel_bar_extent.fits
image_file7 = image_file7[0].data[0][0]
print(image_file7.shape)


image_file8 = fits.open('pf_iuwt_residual.fits')
image_file8 = image_file8[0].data
print(image_file8.shape)

# RMSE3 = np.linalg.norm(image_file8,ord=2)/256
# maxrestore3=np.max(image_file6)
# print(RMSE3,maxrestore3/RMSE3)
# print("max restored",maxrestore3)


image_file9 = fits.open('m51_dec1000_102023-09-13.fits')#condat_deconvolved_bar_extent
image_file9 = image_file9[0].data
print(image_file9.shape)
image_file10 = fits.open('m51_ture1000_102023-09-13.fits')#condat_trueskymodel_bar_extent.fits
image_file10 = image_file10[0].data
print(image_file10.shape)
image_file11 = fits.open('m51_res1000_102023-09-13.fits')
image_file11 = image_file11[0].data
print(image_file11.shape)




image_file0 = 1*np.abs(image_file0-np.min(image_file0))/np.abs(np.max(image_file0)-np.min(image_file0))
image_file1 = 1*np.abs(image_file1-np.min(image_file1))/np.abs(np.max(image_file1)-np.min(image_file1))
image_file2 = 1*np.abs(image_file2-np.min(image_file2))/np.abs(np.max(image_file2)-np.min(image_file2))
err_sky_model0 = np.abs(image_file0 - image_file1)


image_file3 = 1*np.abs(image_file3-np.min(image_file3))/np.abs(np.max(image_file3)-np.min(image_file3))
image_file4 = 1*np.abs(image_file4-np.min(image_file4))/np.abs(np.max(image_file4)-np.min(image_file4))
image_file5 = 1*np.abs(image_file5-np.min(image_file5))/np.abs(np.max(image_file5)-np.min(image_file5))
err_sky_model1 = np.abs(image_file3 - image_file4)




image_file7 = 1*np.abs(image_file7-np.min(image_file7))/np.abs(np.max(image_file7)-np.min(image_file7))
image_file6 = 1*np.abs(image_file6-np.min(image_file6))/np.abs(np.max(image_file6)-np.min(image_file6))
image_file8 = 1*np.abs(image_file8-np.min(image_file8))/np.abs(np.max(image_file8)-np.min(image_file8))
err_sky_model2 = np.abs(image_file6 - image_file7)


image_file10 = 1*np.abs(image_file10-np.min(image_file10))/np.abs(np.max(image_file10)-np.min(image_file10))
image_file9 = 1*np.abs(image_file9-np.min(image_file9))/np.abs(np.max(image_file9)-np.min(image_file9))

image_file11 = 1*np.abs(image_file11-np.min(image_file11))/np.abs(np.max(image_file11)-np.min(image_file11))
err_sky_model3 = np.abs(image_file9 - image_file10)



import numpy

RMSE = numpy.sqrt(np.linalg.norm(image_file2,ord=2)/256)
DR2 = np.max(image_file0) / RMSE
SNR2=20*log(np.std(image_file1)/np.std(err_sky_model0),10)
print("hogbom clean SNR2 and DR2 is :",SNR2,DR2,RMSE)


RMSE = numpy.sqrt(np.linalg.norm(image_file5,ord=2)/256)
DR2 = np.max(image_file3) / RMSE
SNR2=20*log(np.std(image_file4)/np.std(err_sky_model1),10)
print("ms clean SNR and DR is:",SNR2,DR2,RMSE)



RMSE = numpy.sqrt(np.linalg.norm(image_file8,ord=2)/256)
DR2 = np.max(image_file6) / RMSE
SNR2=20*log(np.std(image_file7)/np.std(err_sky_model2),10)
print("iuwt pf SNR and DR is:",SNR2,DR2,RMSE)

RMSE = numpy.sqrt(np.linalg.norm(image_file11,ord=2)/256)
DR2 = np.max(image_file9) / RMSE
SNR2=20*log(np.std(image_file10)/np.std(err_sky_model3),10)
print("iuwt cs SNR and DR is:",SNR2,DR2,RMSE)

