
import scipy.io as sio    

def load_data(num):

    data =  sio.loadmat(r'{num}\train_data_6.mat'.format(num=num))  
    x_train = data['train_data']  
   
    data =  sio.loadmat(r'{num}\train_hot_label.mat'.format(num=num))  
    y_train = data['train_hot_label'] 

    data =  sio.loadmat(r'{num}\test_data_6.mat'.format(num=num))  
    x_test = data['test_data'] 

    data =  sio.loadmat(r'{num}\test_hot_label.mat'.format(num=num))  
    y_test = data['test_hot_label'] 


    return (x_train, y_train), (x_test, y_test)