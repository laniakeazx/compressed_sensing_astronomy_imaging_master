import matplotlib.pyplot as plt
from PIL import Image
from astropy.visualization import astropy_mpl_style

plt.style.use(astropy_mpl_style)
from astropy.io import fits
import numpy as np
import random
from math import log

# image_file1 = get_pkg_data_filename('M51ha.fits')
# fits.info(image_file1)
# image_data1 = fits.getdata(image_file1, ext=0)
# image_data1 = image_data1

image_file1 = fits.open('m51_256.fits')
image_file1 = image_file1[0].data

# image_file1=np.delete(image_file1,1,axis=0)

# grey=fits.PrimaryHDU(image_file1)
# greyHDU=fits.HDUList([grey])
# greyHDU.writeto('m51fits2osm'+'.fits',overwrite=True)

# new_width,new_height=[256,384]
# img = Image.fromarray(image_file1)
# # 前两个坐标点是左上角坐标
# # 后两个坐标点是右下角坐标
# # width在前， height在后
# box = (100, 0, 924, 1023)
# img = img.crop(box)
# new_size = (new_width, new_height)
# resized_img = img.resize(new_size)
# # 将PIL图像对象转换为Numpy数组
# data = np.array(resized_img)
# # 创建FITS文件头
# header = fits.Header()
# # 将Numpy数组保存为FITS数据
# hdu = fits.PrimaryHDU(data, header=header)
# hdu.writeto('example3.fits',overwrite=True)
# resized_img.show()




width,height=image_file1.shape

print(width,height)
print(np.max(image_file1))
print(image_file1[0][1])

# 遍历每一个像素
for x in range(width):
    for y in range(height):
        # 获取像素值
        pixel = image_file1[x][y]
        # 打印像素值和位置
        # print("像素位置：({0}, {1})，像素值：{2}".format(x, y, pixel))
        x1 = float(x)
        y1 = float(y)

        x1 =  0.5 * (x1/25.6)
        # x1 = 0.5 * (x1 / 38.4)
        y1 =  0.5 * (y1/25.6)

        pixel = 5 * (pixel/ 1058.0227)

        x1 = x1 + 17.500 -20
        y1 = y1 - 32.500 +4

        # x1 =  1 * (x1/25.6)
        # y1 =  1 * (y1/25.6)
        #
        # pixel = 5 * (pixel/ 1072.098)
        #
        # x1 = x1 + 15.000
        # y1 = y1- 35.000

        maj_arcsec_max = 0
        maj_arcsec_min = 0
        maj_arcsec_ponit1 = 0
        maj_arcsec = random.uniform(maj_arcsec_max, maj_arcsec_min)
        maj_arcsec_result = round(maj_arcsec)
        # print(maj_arcsec_result)
        # Minor axis FWHM
        min_arcsec_max = 0
        min_arcsec_min = 0
        min_arcsec_ponit1 = 0
        min_arcsec = random.uniform(min_arcsec_max, min_arcsec_min)
        min_arcsec_result = round(min_arcsec)

        location_fux = [x1,y1,pixel*1,0,0,0,100000000.0,-0.7,0.0,maj_arcsec_result,min_arcsec_result,0]
        location_fux = str(location_fux)
        location_fux = location_fux.replace(',', ' ')
        location_fux = location_fux.replace('[', '')
        location_fux = location_fux.replace(']', '')
        with open('data3_m51.txt', 'a+', encoding='utf-8') as f:
            f.write('\n')
            for data in location_fux:
                f.write(data)
            f.close()


image_file2 = fits.open('3c288.fits')
image_file2 = image_file2[0].data
print(image_file2.shape)
#
#
# image_file3 = get_pkg_data_filename('SNR_G55_10s_natural_image.fits')
# fits.info(image_file3)
# image_data3 = fits.getdata(image_file3, ext=0)
# image_data3 = image_data3[0][0]
# print(image_data3.shape)
#
# image_file4= get_pkg_data_filename('SNR_G55_10s_dirty_model.fits')
# fits.info(image_file4)
# image_data4 = fits.getdata(image_file4, ext=0)
# image_data4= image_data4[0][0]
# print(image_data4.shape)
#
# image_file5 = get_pkg_data_filename('SNR_G55_10s_ms_model.fits')
# fits.info(image_file5)
# image_data5 = fits.getdata(image_file5, ext=0)
# image_data5 = image_data5[0][0]
# print(image_data5.shape)
#
# image_file6 = get_pkg_data_filename('SNR_G55_10s_natural_model.fits')
# fits.info(image_file6)
# image_data6 = fits.getdata(image_file6, ext=0)
# image_data6 = image_data6[0][0]
# print(image_data6.shape)
#
# image_file7 = get_pkg_data_filename('SNR_G55_10s_dirty_residual.fits')
# fits.info(image_file7)
# image_data7 = fits.getdata(image_file7, ext=0)
# image_data7 = image_data7[0][0]
# print(image_data7.shape)
#
# image_file8 = get_pkg_data_filename('SNR_G55_10s_ms_residual.fits')
# fits.info(image_file8)
# image_data8 = fits.getdata(image_file8, ext=0)
# image_data8 = image_data8[0][0]
# print(image_data8.shape)
#
# image_file9 = get_pkg_data_filename('SNR_G55_10s_natural_residual.fits')
# fits.info(image_file9)
# image_data9 = fits.getdata(image_file9, ext=0)
# image_data9 = image_data9[0][0]
# print(image_data9.shape)



import numpy as np
from math import log
# image_data1 = 1*np.abs(image_data1-np.min(image_data1))/np.abs(np.max(image_data1)-np.min(image_data1))
# image_data2 = 1*np.abs(image_data2-np.min(image_data2))/np.abs(np.max(image_data2)-np.min(image_data2))
# image_data3 = 0.1*np.abs(image_data3-np.min(image_data3))/np.abs(np.max(image_data3)-np.min(image_data3))
# image_data4 = 1*np.abs(image_data4-np.min(image_data4))/np.abs(np.max(image_data4)-np.min(image_data4))

plt.subplot(331)
plt.imshow(image_file1, cmap='jet')
plt.colorbar()
plt.title("iuwt-cs skymodel")
#
plt.subplot(332)
plt.imshow(image_file2, cmap='jet')
plt.colorbar()
plt.title("iuwt-cs dirty")
#
#
# plt.subplot(333)
# plt.imshow(image_data3, cmap='jet')
# plt.colorbar()
# plt.title("iuwt-cs deconv")
#
#
# plt.subplot(334)
# plt.imshow(image_data4, cmap='jet')
# plt.colorbar()
# plt.title("iuwt-cs res")
#
# plt.subplot(335)
# plt.imshow(image_data5, cmap='jet')
# plt.colorbar()
# plt.title("iuwt-cs res")
#
# plt.subplot(336)
# plt.imshow(image_data6, cmap='jet')
# plt.colorbar()
# plt.title("iuwt-cs res")
#
#
# plt.subplot(337)
# plt.imshow(image_data7, cmap='jet')
# plt.colorbar()
# plt.title("iuwt-cs res")
#
#
#
# plt.subplot(338)
# plt.imshow(image_data8, cmap='jet')
# plt.colorbar()
# plt.title("iuwt-cs res")
#
#
#
# plt.subplot(339)
# plt.imshow(image_data9, cmap='jet')
# plt.colorbar()
# plt.title("iuwt-cs res")
plt.show()
#
#
# SNR = 20*log((np.linalg.norm(image_data4,ord=2))/(np.linalg.norm(image_data3,ord=2)),10)
# DR = np.max(image_data1)/np.std(image_data3)
# print("SNR and DR is :",SNR,DR)
