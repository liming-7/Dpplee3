'''
pytorch example
'''
from __future__ import print_function
from keras.datasets import mnist
from keras.utils import np_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from dpplee3.dpplee3_model import Dpplee3_model
from dpplee3.data_rdd_transform import to_simple_rdd


from pyspark import SparkContext, SparkConf

batch_size=64
test_batch_size=64
epochs=100
nb_classes=10
momentum=0.01
lr=0.01
log_interval=10

# Create Spark context
conf = SparkConf().setAppName('dpplee3').setMaster('local[8]')
sc = SparkContext(conf=conf)
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(60000, 1, 28, 28)
# x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
# x_test = x_test.astype("float32")
x_train /= 255
# x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(y_train)
# Convert class vectors to binary class matrices
# y_train = np_utils.to_categorical(y_train, nb_classes)
# y_test = np_utils.to_categorical(y_test, nb_classes)

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, x_train, y_train)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
# class Net(nn.Module):  
#     def __init__(self):  
#         super(Net, self).__init__()  
#         self.conv1 = nn.Sequential( # (1,28,28)  
#                      nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,  
#                                stride=1, padding=2), # (16,28,28)  
#         # 想要con2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2  
#                      nn.ReLU(),  
#                      nn.MaxPool2d(kernel_size=2) # (16,14,14)  
#                      )  
#         self.conv2 = nn.Sequential( # (16,14,14)  
#                      nn.Conv2d(16, 32, 5, 1, 2), # (32,14,14)  
#                      nn.ReLU(),  
#                      nn.MaxPool2d(2) # (32,7,7)  
#                      )  
#         self.out = nn.Linear(32*7*7, 10)  
  
#     def forward(self, x):  
#         x = self.conv1(x)  
#         x = self.conv2(x)  
#         x = x.view(x.size(0), -1) # 将（batch，32,7,7）展平为（batch，32*7*7）  
#         output = self.out(x)  
#         return output  
model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

spark_model = Dpplee3_model(sc, model, 'epoch', 'nll_loss', 'asynchronous', 2, frequency_num =2)
spark_model.set_worker_optimizer('SGD', lr =0.01, momentum=0.5)
spark_model.set_server_optimizer('SGD', lr =0.01, momentum=0.5)
spark_model.train(rdd, 200, 64)