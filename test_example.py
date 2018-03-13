'''
pytorch example
'''
from __future__ import print_function
from keras.datasets import mnist
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from dpplee3_model import Dpplee3

batch_size=64
test_batch_size=64
epochs=100
momentum=0.01
lr=0.01
log_interval=10

# Create Spark context
conf = SparkConf().setAppName('Mnist_Spark').setMaster('local[8]')
sc = SparkContext(conf=conf)
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, x_train, y_train)

class Net(nn.Module):
    '''define neural network'''
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

model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

spark_model = Dpplee3(sc, model, None, 'SGD', 1, 'asynchronous', 2)
spark_model.train(rdd, 1000, 64)