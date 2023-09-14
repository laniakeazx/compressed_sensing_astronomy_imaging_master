import numpy
import numpy as np
import matplotlib.pyplot as plt
from pysap.plugins.astro.deconvolution.deconvolve import sparse_deconv_condatvu
from modopt.signal.noise import add_noise
from modopt.math.convolve import convolve
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import datetime

def ndarray_con(input_narray):
    print('ndarray type:', type(input_narray))
    print('ndarray len:', len(input_narray))
    print('ndarray的维度: ', input_narray.ndim)
    print('ndarray的形状: ', input_narray.shape)
    print('ndarray的元素数量: ', input_narray.size)
    print('ndarray中的数据类型: ', input_narray.dtype)
    print('ndarray的值',input_narray)

def req_numpyarray(array):
    ymax = 256
    ymin = 0
    xmax = max(map(max, array))
    xmin = min(map(min, array))

    for i in range(256):
        for j in range(256):
            array[i][j] = round(((ymax - ymin) * (array[i][j] - xmin) / (xmax - xmin)) + ymin)

    return array

# galaxy = get_sample_data('astro-fits')
# psf = get_sample_data('astro-psf')
# ndarray_con(galaxy.data)
# print(psf.data[20])
# print('psf max:', np.max(psf.data))
# galaxy = 255*galaxy.data
# psf =255*psf.data
# galaxy = np.insert(galaxy, 1, 0, axis=0)
# galaxy = np.insert(galaxy, 1, 0, axis=1)
# psf = np.insert(psf, 1, 0, axis=0)
# psf = np.insert(psf, 1, 0, axis=1)

dirty_img_data = get_pkg_data_filename("MSclean_20000ni_256256model.fits")#M31
#100points_256_18000_multiscale_64arc_1.5sca3model.fits
# SNR_G55_10s_1000.naturalimage.fits
dirty_img = fits.getdata(dirty_img_data)
dirty_img = numpy.array(dirty_img, dtype=numpy.float64)
# dirty_img = dirty_img/7
dirty_img = (dirty_img-np.min(dirty_img))/(np.max(dirty_img)-np.min(dirty_img))
dirty_img = 1*dirty_img[0][0]#3* #0.001  3 5
# dirty_img = np.insert(dirty_img, 128, 1, axis=0)
# dirty_img = np.insert(dirty_img, 128, 1, axis=1)
# dirty_img = req_numpyarray(dirty_img)
# dirty_img = (dirty_img-np.min(dirty_img))/(np.max(dirty_img)-np.min(dirty_img))
print(np.shape(dirty_img))
# ndarray_con(dirty_img)
print('dirty_img max:', np.max(dirty_img))
psf_img_data = get_pkg_data_filename("MSclean_20000ni_256256psf.fits")#m51_256_16000_multiscale_64arc_3.5sca3psf m51_256_16000_multiscale_64arc_3.5sca3psf.fits
#m51_256_20000_multiscale_65arc_3scapsf.fits
psf_img = fits.getdata(psf_img_data)
psf_img = numpy.array(psf_img, dtype=numpy.float64)
psf_img = 0.015*psf_img[0][0]#0.001 #0.015 0.015

# psf_img = np.insert(psf_img, 128, 1, axis=0)
# psf_img = np.insert(psf_img, 128, 1, axis=1)
# for i in range(psf_img.shape[0]):
#     col = psf_img[:,i]
#     col[np.isnan(col)] = 0
# psf_img = np.nan_to_num(psf_img)
# psf_img[np.isnan(psf_img)] = 0
# # psf_img = 0.1*(psf_img-np.min(psf_img))/(np.max(psf_img)-np.min(psf_img))
# # psf_img = req_numpyarray(psf_img)

ndarray_con(psf_img)
print('psf max:', np.max(psf_img))
print('dirty_img max:', np.max(dirty_img))
print('psf_img max:', np.max(psf_img))
obs_data = add_noise(convolve(dirty_img, psf_img), sigma=0.00001)#0.0005
ndarray_con(obs_data)
# obs_data=30*obs_data
print('obs_data max:', np.max(obs_data))
# noise = 4*np.random.rand(256,256)
# obs_data = noise +obs_data
import time
#格式化时间戳
algorithm_starttime=time.time()#获得时间元组
deconv_data = sparse_deconv_condatvu(1*obs_data, psf_img, n_iter=1000,n_reweights=10)
algorithm_endtime=time.time()#获得时间元组
algorithm_spendtime=algorithm_endtime-algorithm_starttime
print("algorithm time spend:",algorithm_spendtime)
# deconv_data = sparse_deconv_condatvu(100*obs_data, psf_img, n_iter=100,n_reweights=1)
#
from math import log,sqrt
residual2 = np.abs(obs_data - deconv_data)
RMSE1 = np.linalg.norm(residual2,ord=2)/256
maxrestore1=np.max(deconv_data)
print("max restored",maxrestore1)
print(RMSE1,maxrestore1/RMSE1)
deconv_data = 1*np.abs(deconv_data-np.min(deconv_data))/np.abs(np.max(deconv_data)-np.min(deconv_data))
dirty_img = 1*np.abs(dirty_img-np.min(dirty_img))/np.abs(np.max(dirty_img)-np.min(dirty_img))
err_sky_model1 = np.abs(deconv_data - dirty_img)
err_sky_model1 = 1*np.abs(err_sky_model1-np.min(err_sky_model1))/np.abs(np.max(err_sky_model1)-np.min(err_sky_model1))
SNR = 20*log((np.linalg.norm(dirty_img,ord=2))/(np.linalg.norm(err_sky_model1,ord=2)),10)
deconv_data2=deconv_data

deconv_data2 = add_noise(convolve(deconv_data2, psf_img), sigma=0.00001)#0.0005

obs_data = 1*np.abs(obs_data-np.min(obs_data))/np.abs(np.max(obs_data)-np.min(obs_data))
deconv_data2 = 1*np.abs(deconv_data2-np.min(deconv_data2))/np.abs(np.max(deconv_data2)-np.min(deconv_data2))
residual = np.abs(obs_data - deconv_data2)#-0.3
print('residual max:', np.max(residual))
residual2 = np.abs(deconv_data-dirty_img)
residual2 = 1*np.abs(residual2-np.min(residual2))/np.abs(np.max(residual2)-np.min(residual2))
print('residual2 max:', np.max(residual2))
# residual = np.abs((dirty_img - deconv_data))
# obs_data = add_noise(convolve(galaxy, psf), sigma=0.0005)
# deconv_data = sparse_deconv_condatvu(obs_data, psf, n_iter=100,n_reweights=1)
# residual = np.abs(galaxy - deconv_data)

datetime_object = datetime.datetime.now()
datetime_object = str(datetime_object)[0:10]
print(datetime_object)
ni_and_reni = "1000_10"
plt.subplot(335)
plt.imshow(psf_img, cmap='jet')
plt.colorbar()
plt.title('the psf image ')

plt.subplot(331)
plt.imshow(dirty_img, cmap='jet')
plt.colorbar()
plt.title('the ture galaxy image ')
# plt.axis([0,255,0,255])
grey=fits.PrimaryHDU(dirty_img)
greyHDU=fits.HDUList([grey])
greyHDU.writeto('m51_ture'+ni_and_reni+datetime_object+'.fits',overwrite=True)

plt.subplot(332)
plt.imshow(obs_data, cmap='jet')
plt.colorbar()
plt.title('the dirty galaxy image')
# plt.axis([0,255,0,255])
grey=fits.PrimaryHDU(obs_data)
greyHDU=fits.HDUList([grey])
greyHDU.writeto('m51_dirty'+ni_and_reni+datetime_object+'.fits',overwrite=True)

plt.subplot(333)
plt.imshow(deconv_data, cmap='jet')
plt.colorbar()
plt.title('deconvolved galaxy image')
# plt.axis([0,255,0,255])
grey=fits.PrimaryHDU(deconv_data)
greyHDU=fits.HDUList([grey])
greyHDU.writeto('m51_dec'+ni_and_reni+datetime_object+'.fits',overwrite=True)


plt.subplot(334)
plt.imshow(residual, cmap='jet')
plt.colorbar()
plt.title('residual galaxy image')
# plt.axis([0,255,0,255])
grey=fits.PrimaryHDU(residual)
greyHDU=fits.HDUList([grey])
greyHDU.writeto('m51_res'+ni_and_reni+datetime_object+'.fits',overwrite=True)



plt.subplot(336)
plt.imshow(err_sky_model1, cmap='jet')
plt.colorbar()
plt.title(' residual image2')
# plt.axis([0,255,0,255])
grey=fits.PrimaryHDU(err_sky_model1)
greyHDU=fits.HDUList([grey])
greyHDU.writeto('m51_error'+ni_and_reni+datetime_object+'.fits',overwrite=True)
plt.show()


import numpy as np
from math import log,sqrt



dirty_img = 1*np.abs(dirty_img-np.min(dirty_img))/np.abs(np.max(dirty_img)-np.min(dirty_img))
residual = 1*np.abs(residual-np.min(residual))/np.abs(np.max(residual)-np.min(residual))
deconv_data = 1*np.abs(deconv_data-np.min(deconv_data))/np.abs(np.max(deconv_data)-np.min(deconv_data))
# SNR=20*log((np.linalg.norm(dirty_img, ord=2))/(np.linalg.norm(residual, ord=2)),10)
#
# DR=np.max(deconv_data)/np.std(residual)

error_sky_model3 = np.abs(deconv_data - dirty_img )
RMSE3 = np.linalg.norm(residual2,ord=2)/256
DR3= maxrestore1 / RMSE3
SNR3 = 20*log((np.std(deconv_data))/np.std(error_sky_model3),10)

print("rmse3,dr3,snr3",RMSE3,DR3,SNR3)
