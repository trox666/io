import os
import  sys 
import  pickle 
import  numpy as  np 

def  unpickle (file):  
    fp = open(file, 'rb') 
    if  sys.version_info.major == 2: 
        data = pickle.load(fp) 
    elif  sys.version_info.major == 3: 
        data = pickle.load(fp, encoding='latin-1') 
        fp.close() 
    return  data 

def load_CIFAR10(cifar10dir):
    """load_CIFAR10; returns a Nx3x32x32 uint8 numpy array from the 
                     folder containing the batches of the CIFAR 10 dataset
    """
    X_train = None
    y_train = []
    for  i in  range(1, 6):
        data_dic = unpickle(cifar10dir+"/data_batch_{}". format(i))
        if  i == 1:
            X_train = data_dic['data']
        else:
            X_train = np. vstack((X_train, data_dic['data']))
            y_train += data_dic['labels']
    test_data_dic = unpickle(cifar10dir + "/test_batch")
    X_test = test_data_dic['data']
    X_test = X_test.reshape(len(X_test), 3, 32, 32)
    y_test = np. array(test_data_dic['labels'])
    X_train = X_train.reshape((len(X_train), 3, 32, 32))
    y_train = np. array(y_train)
    return X_train, y_train, X_test, y_test


def get_cifar10(folder):
    '''get_cifar10; loads the batches in CIFAR 10 folder, returns a Nx3072 
                    numpy array.
    32x32x3
    '''
    tr_data = np.empty((0,32*32*3))
    tr_labels = np.empty(1)
    for i in range(1,6):
        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            tr_data = data_dict['data']
            tr_labels = data_dict['labels']
        else:
            tr_data = np.vstack((tr_data, data_dict['data']))
            tr_labels = np.hstack((tr_labels, data_dict['labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    te_data = data_dict['data']
    te_labels = np.array(data_dict['labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    label_names = bm['label_names']
    return tr_data, tr_labels, te_data, te_labels, label_names   
