from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

def get_files(root):
  return [os.path.join(root, n)
    for n in sorted(os.listdir(root))
    if os.path.isfile(os.path.join(root, n))
  ]

sample_size = 5
for i in np.arange(sample_size):
    display(Image(filename=np.random.choice(get_files('/home/rohith/Udacity/deep-learning/notMNIST_small/B')))) #Not sure if this will work with regular python
    #display(Image(filename=np.random.choice(get_files(np.random.choice(train_folders)))))

image_size = 28 #28x28
pixels = 255.0

image_name = np.random.choice(get_files('/home/rohith/Udacity/deep-learning/notMNIST_small/D'))
#image_name = np.random.choice(get_files(np.random.choice(train_folders)))
image = ndimage.imread(image_name).astype('float') #Reading image as float
print(type(image))
print(image.shape)

plt.imshow(image)
plt.show() #image will look different because of pyplot

'''
## Plot image using a scatterplot

colors = [str(i/pixel_depth) for i in np.ravel(image_data)]
plt.scatter(
  np.tile(np.arange(image_size), image_size),
  np.repeat(np.flipud(np.arange(image_size)), image_size),
  s=150,
  c=colors,
  marker='s'
)
plt.show()

## Plot image using a scatterplot by setting cmap option

colors = [str(i/pixel_depth) for i in np.ravel(image_data)]
plt.scatter(
  np.tile(np.arange(image_size), image_size),
  np.repeat(np.flipud(np.arange(image_size)), image_size),
  s=150,
  c=colors,
  marker='s',
  cmap=plt.cm.winter
)
plt.show()
'''