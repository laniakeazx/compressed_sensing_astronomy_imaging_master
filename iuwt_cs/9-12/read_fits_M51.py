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

image_file3 = fits.open('MSclean_25000ni_256256_thre1mjy_natural_9_13image.fits')#m51_256_15000_multiscale_64arc_3sca3image
image_file3 = image_file3[0].data[0][0]
print(image_file3.shape)
# image_file3 = 1*np.abs(image_file3-np.min(image_file3))/np.abs(np.max(image_file3)-np.min(image_file3))

image_file4 = fits.open('MSclean_25000ni_256256_thre1mjy_natural_9_13model.fits')#m51_256_15000_multiscale_64arc_3sca3model
image_file4 = image_file4[0].data[0][0]
print(image_file4.shape)

image_file5 = fits.open('MSclean_25000ni_256256_thre1mjy_natural_9_13residual.fits')#m51_256_15000_multiscale_64arc_3sca3residual
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


image_file9 = fits.open('m51_dec_1000_62023-09-12.fits')#condat_deconvolved_bar_extent
image_file9 = image_file9[0].data
print(image_file9.shape)
image_file10 = fits.open('m51_ture_1000_62023-09-12.fits')#condat_trueskymodel_bar_extent.fits
image_file10 = image_file10[0].data
print(image_file10.shape)
image_file11 = fits.open('m51_res_1000_62023-09-12.fits')
image_file11 = image_file11[0].data
print(image_file11.shape)



# RMSE4 = np.linalg.norm(image_file11,ord=2)/256
# maxrestore4=np.max(image_file9)
# print(RMSE4,maxrestore4/RMSE4)
#
# print("max restored",maxrestore4)
# b = np.ones(3)
# print(b)
# array([1., 1., 1.])
# np.insert(a,0,b,axis=1)


# image_data1 = 1*np.abs(image_data1-np.min(image_data1))/np.abs(np.max(image_data1)-np.min(image_data1))
# image_data2 = 1*np.abs(image_data2-np.min(image_data2))/np.abs(np.max(image_data2)-np.min(image_data2))
# image_data3 = 0.1*np.abs(image_data3-np.min(image_data3))/np.abs(np.max(image_data3)-np.min(image_data3))
# image_file3 = 1*np.abs(image_file3-np.min(image_file3))/np.abs(np.max(image_file3)-np.min(image_file3))


import matplotlib.ticker as ticker


# plt.figure()
ax = plt.gca()
# im = ax.imshow(np.arange(100).reshape((10,10)))

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)



norm1 = matplotlib.colors.Normalize(vmin=0,vmax=1.0)  # 设置colorbar显示的最大最小值
norm2 = matplotlib.colors.Normalize(vmin=0,vmax=0.5)  # 设置colorbar显示的最大最小值
norm3 = matplotlib.colors.Normalize(vmin=0,vmax=1.0)  # 设置colorbar显示的最大最小值
norm4 = matplotlib.colors.Normalize(vmin=0,vmax=0.6)  # 设置colorbar显示的最大最小值
norm5 = matplotlib.colors.Normalize(vmin=0,vmax=0.5)  # 设置colorbar显示的最大最小值

image_file0 = 1*np.abs(image_file0-np.min(image_file0))/np.abs(np.max(image_file0)-np.min(image_file0))

image_file0=np.rot90(image_file0)
# image_file0=np.rot90(image_file0)
image_file0 = image_file0[::-1]
plt.subplot(441)
plt.imshow(image_file0, cmap='jet',norm=norm3)
# plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
plt.xticks([])
plt.yticks([])
plt.axis('off')


image_file3 = 1*np.abs(image_file3-np.min(image_file3))/np.abs(np.max(image_file3)-np.min(image_file3))

image_file3=np.rot90(image_file3)
# image_file3=np.rot90(image_file3)
image_file3 = image_file3[::-1]
# plt.title("clean restored")
# image_file1=np.rot90(image_file1)
# image_file1 = image_file1[::-1]
plt.subplot(442)
# image_file1 = image_file1 + 1
# image_file1=np.log(image_file1)
plt.imshow(image_file3,cmap='jet',norm=norm3)#
# cloarbar1=plt.colorbar(extend='both')
# cloarbar1.set_ticks([np.min(image_file1), np.max(image_file1)])
# plt.axis([0,255,0,255])
plt.xticks([])
plt.yticks([])
plt.axis('off')
# plt.title("clean skymodel")


# image_file2[::] = [[row[i] for row in image_file2] for i in range(len(image_file2[0]))][::-1]

# image_file2=np.rot90(image_file2)
# image_file2 = image_file2[::-1]


image_file6 = 1*np.abs(image_file6-np.min(image_file6))/np.abs(np.max(image_file6)-np.min(image_file6))

image_file6=np.rot90(image_file6)
# image_file6=np.rot90(image_file6)
image_file6 = image_file6[::-1]
plt.subplot(4,4,3)
plt.imshow(image_file6,cmap='jet',norm=norm3)#
# cloarbar2=plt.colorbar(extend='both')
# cloarbar2.set_ticks([np.min(image_file2), np.max(image_file2)])
# plt.axis([0,255,0,255])


plt.xticks([])
plt.yticks([])
plt.axis('off')
# plt.title("clean residual")

# image_file2[::] = [[row[i] for row in image_file2] for i in range(len(image_file2[0]))][::-1]

# obs_data1=np.rot90(obs_data1)
# obs_data1 = obs_data1[::-1]
# plt.subplot(531)
# plt.imshow(obs_data1,cmap='jet',norm=norm2)#
#
# # cloarbar2_344=plt.colorbar()
# # cloarbar2.set_ticks([np.min(image_file2), np.max(image_file2)])
# # plt.axis([0,255,0,255])
#
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
#
# # plt.title("clean residual")

image_file9 = 1*np.abs(image_file9-np.min(image_file9))/np.abs(np.max(image_file9)-np.min(image_file9))

image_file9=np.rot90(image_file9)
# image_file9=np.rot90(image_file9)
image_file9 = image_file9[::-1]
plt.subplot(444)
# image_file9 = image_file9 + 1
# image_file9=np.log(image_file9)
plt.imshow(image_file9, cmap='jet',norm=norm3)
cloarbar3 = plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
# cloarbar3.set_ticks([np.min(image_file3), 0.25,0.5, 0.75, np.max(image_file3)])
# plt.tick_params(axis='x', rotation=90)  # 设置x轴标签旋转角度
plt.xticks([])
plt.yticks([])
plt.axis('off')
# plt.title("ms-clean restored")


image_file1 = 1*np.abs(image_file1-np.min(image_file1))/np.abs(np.max(image_file1)-np.min(image_file1))
image_file1=np.rot90(image_file1)
image_file1 = image_file1[::-1]
plt.subplot(445)
# image_file1 = image_file1 + 1
# image_file1=np.log(image_file1)
plt.imshow(image_file1, cmap='jet',norm=norm1)
# plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
plt.xticks([])
plt.yticks([])
plt.axis('off')
# plt.title("ms-clean skymodel")

# image_file5=np.rot90(image_file5)
# image_file5 = image_file5[::-1]
# # image_file5 = image_file5[::-1]
# plt.subplot(4,4,6)
# plt.imshow(image_file5, cmap='jet')
# cloarbar10= plt.colorbar(extend='both')
# # cloarbar10.set_ticks([0.00, 0.10,0.20, 0.30,0.40, 0.50])
# # plt.axis([0,255,0,255])
#
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# # plt.title("ms-clean sresidual")


# obs_data1=np.rot90(obs_data1)
# obs_data1 = obs_data1[::-1]
# image_file5 = image_file5[::-1]

# plt.subplot(532)
# plt.imshow(obs_data1, cmap='jet',norm=norm2)
# # plt.colorbar()
# # plt.axis([0,255,0,255])
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')

# plt.show()






image_file4 = 1*np.abs(image_file4-np.min(image_file4))/np.abs(np.max(image_file4)-np.min(image_file4))
# image_file6=np.rot90(image_file6)
# image_file6=np.rot90(image_file6)
image_file4=np.rot90(image_file4)
image_file4 = image_file4[::-1]
plt.subplot(446)
plt.imshow(image_file4, cmap='jet',norm=norm1)

# cloarbar4=plt.colorbar(extend='both')
# cloarbar4.set_ticks([0.00, 0.20,0.40, 0.60,0.80, 1.00])
plt.xticks([])
plt.yticks([])
plt.axis('off')
# plt.clim(0, 1)
# plt.title("iuwt-cs res")

image_file7 = 1*np.abs(image_file7-np.min(image_file7))/np.abs(np.max(image_file7)-np.min(image_file7))
# image_file7=np.rot90(image_file7)
# image_file7=np.rot90(image_file7)
image_file7=np.rot90(image_file7)
image_file7 = image_file7[::-1]

# image_file7 = image_file7 + 1
# image_file7=np.log(image_file7)

plt.subplot(4,4,7)
plt.imshow(image_file7, cmap='jet',norm=norm1)
# cloarbar8=plt.colorbar(extend='both')
# cloarbar8.set_ticks([0.00, 0.20,0.40, 0.60,0.80, 1.00])
plt.xticks([])
plt.yticks([])
plt.axis('off')
# plt.title("iuwt-cs res")

image_file10 = 1*np.abs(image_file10-np.min(image_file10))/np.abs(np.max(image_file10)-np.min(image_file10))
# image_file8=np.rot90(image_file8)
image_file10=np.rot90(image_file10)
# image_file10=np.rot90(image_file10)
image_file10 = image_file10[::-1]

# image_file8 = image_file8 + 1
# image_file8=np.log(image_file8)
plt.subplot(4,4,8)
plt.imshow(image_file10, cmap='jet',norm=norm1)
cloarbar12=plt.colorbar(extend='both')

# plt.clim(0, 0.4)
plt.xticks([])
plt.yticks([])
plt.axis('off')

# plt.title("iuwt-cs res")
# obs_data3=np.rot90(obs_data3)
# obs_data3 = obs_data3[::-1]
# plt.subplot(5,3,3)
# plt.imshow(obs_data3, cmap='jet',norm=norm2)
# plt.colorbar(extend='both')
# # plt.axis([0,255,0,255])
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')

# err_sky_model1 = err_sky_model1[::-1]


image_file2=np.rot90(image_file2)
image_file2 = image_file2[::-1]
plt.subplot(4,4,9)
plt.imshow(image_file2, cmap='jet')
plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
plt.clim(0, 0.6)
plt.xticks([])
plt.yticks([])
plt.axis('off')

# image_file3 = image_file3/13


image_file5=np.rot90(image_file5)
image_file5 = image_file5[::-1]
# err_sky_model1 = err_sky_model1[::-1]
plt.subplot(4,4,10)
plt.imshow(image_file5, cmap='jet')
plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
plt.clim(0, 0.3)
plt.xticks([])
plt.yticks([])
plt.axis('off')


image_file8=np.rot90(image_file8)
image_file8 = image_file8[::-1]
# err_sky_model1 = err_sky_model1[::-1]
plt.subplot(4,4,11)
plt.imshow(image_file8, cmap='jet')
cloarbar15=plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
plt.clim(0, 0.3)
plt.xticks([])
plt.yticks([])
plt.axis('off')
# font = {'family' : 'serif',
#         'color'  : 'black',
#         'weight' : 'normal',
#         'size'   : 16,
#         }
# cloarbar15.set_label('Brightness(Jy/beam)',fontdict=font) #设置colorbar的标签字体及其大小'Brightness(Jy/beam)'
# # plt.xlabel("c")



image_file11=np.rot90(image_file11)
image_file11 = image_file11[::-1]
plt.subplot(4,4,12)
plt.imshow(image_file11, cmap='jet')
plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
plt.clim(0, 0.08)
plt.xticks([])
plt.yticks([])
plt.axis('off')


image_file0 = 1*np.abs(image_file0-np.min(image_file0))/np.abs(np.max(image_file0)-np.min(image_file0))
image_file1 = 1*np.abs(image_file1-np.min(image_file1))/np.abs(np.max(image_file1)-np.min(image_file1))
err_sky_model0 = np.abs(image_file0 - image_file1)

# err_sky_model0=np.rot90(err_sky_model0)
# err_sky_model0 = err_sky_model0[::-1]
plt.subplot(4,4,13)
plt.imshow(err_sky_model0, cmap='jet',norm=norm5)
# plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
plt.xticks([])
plt.yticks([])
plt.axis('off')

image_file3 = 1*np.abs(image_file3-np.min(image_file3))/np.abs(np.max(image_file3)-np.min(image_file3))
image_file4 = 1*np.abs(image_file4-np.min(image_file4))/np.abs(np.max(image_file4)-np.min(image_file4))
err_sky_model1 = np.abs(image_file3 - image_file4)

# err_sky_model1=np.rot90(err_sky_model1)
# err_sky_model1=np.rot90(err_sky_model1)
# err_sky_model1 = err_sky_model1[::-1]
plt.subplot(4,4,14)
plt.imshow(err_sky_model1, cmap='jet',norm=norm5)
# cloarbar11=plt.colorbar(extend='both')

# plt.axis([0,255,0,255])
plt.xticks([])
plt.yticks([])
plt.axis('off')


image_file7 = 1*np.abs(image_file7-np.min(image_file7))/np.abs(np.max(image_file7)-np.min(image_file7))
image_file6 = 1*np.abs(image_file6-np.min(image_file6))/np.abs(np.max(image_file6)-np.min(image_file6))
err_sky_model2 = np.abs(image_file6 - image_file7)
# # err_sky_model6=np.rot90(err_sky_model6)
# err_sky_model6 = err_sky_model6[::-1]
plt.subplot(4,4,15)
plt.imshow(err_sky_model2, cmap='jet',norm=norm5)
# plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
plt.xticks([])
plt.yticks([])
plt.axis('off')


# err_sky_mode3 = fits.open('m51_256_final.fits')
# err_sky_mode3 = err_sky_mode3[0].data
# print(err_sky_mode3.shape)

image_file10 = 1*np.abs(image_file10-np.min(image_file10))/np.abs(np.max(image_file10)-np.min(image_file10))
image_file9 = 1*np.abs(image_file9-np.min(image_file9))/np.abs(np.max(image_file9)-np.min(image_file9))
err_sky_mode3 = np.abs(image_file9 - image_file10)
# err_sky_model6=np.rot90(err_sky_model6)
# err_sky_model6 = err_sky_model6[::-1]
# err_sky_mode3=np.rot90(err_sky_mode3)
# err_sky_mode3=np.rot90(err_sky_mode3)
# err_sky_mode3=np.rot90(err_sky_mode3)
# err_sky_mode3 = err_sky_mode3[::-1]
plt.subplot(4,4,16)
plt.imshow(err_sky_mode3, cmap='jet',norm=norm5)
plt.colorbar(extend='both')
# plt.axis([0,255,0,255])
plt.xticks([])
plt.yticks([])
plt.axis('off')


# err_sky_model1 = 1*np.abs(err_sky_model1-np.min(err_sky_model1))/np.abs(np.max(err_sky_model1)-np.min(err_sky_model1))
# err_sky_model2 = 1*np.abs(err_sky_model2-np.min(err_sky_model2))/np.abs(np.max(err_sky_model2)-np.min(err_sky_model2))
# err_sky_model3 = 1*np.abs(err_sky_model3-np.min(err_sky_model3))/np.abs(np.max(err_sky_model3)-np.min(err_sky_model3))
# err_sky_model6 = 1*np.abs(err_sky_model6-np.min(err_sky_model6))/np.abs(np.max(err_sky_model6)-np.min(err_sky_model6))

# SNR = 20*log((np.linalg.norm(3*image_file1,ord=2))/(np.linalg.norm(err_sky_model1,ord=2)),10)
# DR = np.max(10*image_file0)/np.std(image_file2)
# print("hogbom clean SNR and DR is :",SNR,DR)
#
# SNR2=20*log(np.std(3*image_file1)/np.std(err_sky_model1),10)
# print(SNR2)
#
# SNR = 20*log((np.linalg.norm(3*image_file4,ord=2))/(np.linalg.norm(err_sky_model2,ord=2)),10)
# DR = np.max(10*image_file3)/np.std(image_file5)
# print("ms clean SNR and DR is :",SNR,DR)
# SNR2=20*log(np.std(3*image_file4)/np.std(err_sky_model2),10)
# print(SNR2)
# SNR = 20*log((np.linalg.norm(3*image_file7,ord=2))/(np.linalg.norm(err_sky_model3,ord=2)),10)
# DR = np.max(10*image_file6)/np.std(image_file8)
# print("iuwt pf SNR and DR is :",SNR,DR)
# SNR2=20*log((np.std(3*image_file7)/np.std(err_sky_model3)),10)
# print(SNR2)
#
# SNR = 20*log((np.linalg.norm(3*image_file10,ord=2))/(np.linalg.norm(err_sky_model6,ord=2)),10)
# DR = np.max(10*image_file9)/np.std(image_file11)
# print("iuwt cs SNR and DR is :",SNR,DR)
# SNR2=20*log((np.std(3*image_file10)/np.std(err_sky_model6)),10)
# print(SNR2)


plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.0,hspace=0.0)
plt.tight_layout()  # 调整整体空白
plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
plt.show()
#
#
# SNR = 20*log((np.linalg.norm(image_data4,ord=2))/(np.linalg.norm(image_data3,ord=2)),10)
# DR = np.max(image_data1)/np.std(image_data3)
# print("SNR and DR is :",SNR,DR)
