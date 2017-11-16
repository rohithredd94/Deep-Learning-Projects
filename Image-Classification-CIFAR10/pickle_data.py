import numpy as np
import pickle
import sys

def load_batch(file):
    with open(file, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')
        x = dataset['data']
        y = dataset['labels']
        x = x.astype(float)
        y = np.array(y)
    return x, y

def reshape_data(data_dict):
    im_tr = np.array(data_dict['train_data'])
    im_tr = np.reshape(im_tr, (-1, 3, 32, 32))
    im_tr = np.transpose(im_tr, (0,2,3,1))
    data_dict['train_data'] = im_tr
    im_te = np.array(data_dict['test_data'])
    im_te = np.reshape(im_te, (-1, 3, 32, 32))
    im_te = np.transpose(im_te, (0,2,3,1))
    data_dict['test_data'] = im_te
    return data_dict

def load_data():
    data = []
    labels = []
    for i in range(1, 6):
        file = 'data/data_batch_'+str(i)
        data_i, labels_i = load_batch(file)
        data.append(data_i)
        labels.append(labels_i)

    train_data = np.concatenate(data)
    train_labels = np.concatenate(labels)
    del data, labels
    
    #print(train_data.shape, train_labels.shape)
    
    test_data, test_labels = load_batch('data/test_batch')
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck']

    #Normalize train data
    mean_data = np.mean(train_data, axis = 0)
    train_data -= mean_data
    test_data -= mean_data

    final_data = {
        'train_data' : train_data,
        'train_labels' : train_labels,
        'test_data' : test_data,
        'test_labels' : test_labels,
        'classes' : classes
    }
    #final_data = reshape_data(final_data)
    return final_data
    

if __name__ == '__main__':
    data = load_data()
    print(data['train_data'].shape)
    print(data['train_labels'].shape)
    print(data['test_data'].shape)
    print(data['test_labels'].shape)

