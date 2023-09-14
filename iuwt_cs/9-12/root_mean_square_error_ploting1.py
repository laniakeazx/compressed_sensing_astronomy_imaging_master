import matplotlib.pyplot as plt

from astropy.visualization import astropy_mpl_style

# plt.style.use(astropy_mpl_style)
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from math import log
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
image_file0 = fits.open('m51_ture1000_102023-09-13.fits')
image_file0 = image_file0[0].data
# image_file0 = image_file0[0].data[0][0]
# image_file0=np.delete(image_file0,1,axis=0)
print(image_file0.shape)
image_file1 = fits.open('m51_dec1000_102023-09-13.fits')
# image_file1 = image_file1[0].data
image_file1 = 1*image_file1[0].data
# image_file0=np.delete(image_file0,1,axis=0)
print(image_file1.shape)
image_file1 = 1*np.abs(image_file1-np.min(image_file1))/np.abs(np.max(image_file1)-np.min(image_file1))
image_file0 = 1*np.abs(image_file0-np.min(image_file0))/np.abs(np.max(image_file0)-np.min(image_file0))
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 22:21:12 2018
@author: Raunak Goswami
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# reading the data
"""
here the directory of my code and the headbrain6.csv file 
is same make sure both the files are stored in same folder or directory
"""
x = image_file0
y = image_file0
y1 = image_file1
# splitting the data into training and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=0)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y1, test_size=1 / 4, random_state=0)

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor2 = LinearRegression()
regressor.fit(x_train, y_train)
regressor2.fit(x_train2, y_train2)
# predict the test result
y_pred = regressor.predict(x_test)
y_pred2 = regressor2.predict(x_test2)
# to see the relationship between the training data values
plt.scatter(x_train, y_train, c='darkorange',marker='+',label='True sky',s=26)
plt.scatter(x_train2, y_train2, c='steelblue',marker='.',label='IUWT-FISTA',s=26)
plt.tick_params(labelsize=16)
plt.legend()
plt.show()
# to see the relationship between the predicted
# brain weight values using scattered graph
# plt.plot(x_test, y_pred)
plt.scatter(x_test, y_test, c='red',marker='.',label='decon')
plt.xlabel('headsize')
plt.ylabel('brain weight')

# errorin each value
for i in range(0, 60):
    print("Error in value number", i, (y_test[i] - y_pred[i]))
time.sleep(1)

# combined rmse value
rss = ((y_test - y_pred) ** 2).sum()
mse = np.mean((y_test - y_pred) ** 2)
print("Final rmse value is =", np.sqrt(np.mean((y_test - y_pred) ** 2)))


