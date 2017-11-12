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

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    print(num_images)
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000) #Names of pickle files are stored in the train_datasets
test_datasets = maybe_pickle(test_folders, 1800)

#Show some random pickle data - Problem 2
data = np.load('./notMNIST_large/D.pickle')
print(data.shape)
plt.imshow(data[3,:,:])
plt.show()

#Check if balanced or not - Problem 3
#Checking balance for train data
train_data = len(train_folders)
no_of_files = np.empty(shape=train_data, dtype=np.int64)

for i in np.arange(train_data):
    data = np.load(str(train_folders[i])+'.pickle')
    print(data.shape) #(no.of images, width, height) - no.of images should be similar across all classes

'''
train_stats_perc = 100*train_stats/np.float32(sum(train_stats))

train_stats_perc
'''

#Checking balance for test data
test_data = len(test_folders)
no_of_files = np.empty(shape=test_data, dtype=np.int64)

for i in np.arange(test_data):
    data = np.load(str(test_folders[i])+'.pickle')
    print(data.shape) #(no.of images, width, height) - no.of images should be similar across all classes

'''
test_stats_perc = 100*test_stats/np.float32(sum(test_stats))

test_stats_perc
'''