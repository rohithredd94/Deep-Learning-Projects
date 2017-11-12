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

#Finding near duplicates using image hashing - This code is NOT WORKING - Problem 5

def unnormalize(image, pixel_depth):
  return (pixel_depth*image+pixel_depth/2).astype(np.uint8)

# The input argument hsize is the hash size
def image_hash(image, hsize=8):
  # Grayscale and shrink the image
  icon = PIL.Image.fromarray(image).convert('L').resize((hsize+1, hsize), PIL.Image.ANTIALIAS)
  icon = np.array(icon)

  # Compare intensity values of adjacent pixels row-wise
  diff = np.empty([hsize, hsize], dtype=np.bool_)
  for row in np.arange(hsize):
    for col in np.arange(hsize):
      diff[row, col] = icon[row, col] > icon[row, col+1]

  # Convert binary vector to hexadecimal string
  hexadecimal = np.empty(hsize, dtype=np.dtype(bytes, hsize/4))
  for i in np.arange(hsize):
    hexadecimal[i] = \
      hex(int(''.join(str(b) for b in np.flipud(diff[i, :].astype(int))), 2))[2:].rjust(2, '0')
    
  return ''.join(str(hexadecimal))
hash_size = 8
traindata_hash = np.empty(train_size, dtype=np.dtype(bytes,(hash_size**2)/4))
print(np.arange(train_size))
for i in np.arange(train_size):
    traindata_hash[i] = image_hash(unnormalize(train_dataset[i,:,:], pixel_depth))

print(traindata_hash[0])

validdata_hash = np.empty(valid_size, dtype=np.dtype(bytes,(hash_size**2)/4))
print(np.arange(valid_size))
for i in np.arange(valid_size):
    validdata_hash[i] = image_hash(unnormalize(valid_dataset[i,:,:], pixel_depth))

print(validdata_hash[0])

testdata_hash = np.empty(test_size, dtype=np.dtype(bytes,(hash_size**2)/4))
print(np.arange(test_size))
for i in np.arange(test_size):
    testdata_hash[i] = image_hash(unnormalize(test_dataset[i,:,:], pixel_depth))

print(testdata_hash[0])

unique_train_hashes, unique_train_locs = np.unique(traindata_hash, return_index=True)
print('Number of images in train dataset: %d' % train_size)
print(
  'Number of images in train dataset after excluding near-duplicates: %d'
  % np.size(unique_train_locs)
)
print(
  '%.2f%% of images in train dataset kept' 
  % round(100*np.float32(np.size(unique_train_locs))/train_size, 2)
)