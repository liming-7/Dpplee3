'''
define the worker function
'''
from __future__ import absolute_import
import six.moves.cPickle as pickle
from six.moves import range

class AsynchronousWorker(object):
    '''
    distribute to spark worker by mapPartitions, works on spark worker
    '''
    def __init__(serialized_network, frequency, master_url, master_optimizer, train_config, optimizer_config):
        self.serialized_network = serialized_network
        self.frequency = frequency
        self.master_url = master_url
        self.master_optimizer = master_optimizer
        self.train_config = train_config
        self.optimizer_config = optimizer_config
        self.get_post = Get_post_state_dict(master_url)

    def train(self, data_iterator):
        '''
        Train a keras model on a worker and send asynchronous updates
        to parameter server
        '''
        data, label = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in data_iterator])
        y_train = np.asarray([y for x, y in label_iterator)

        if x_train.size == 0:
            return

        model = pickle_loads(serialized_network)
        epoch_num = self.train_config['epoch']
        batch_size = self.train_config['batch_size']
        sample_num = x_train.shape[0]
        batch_num = int(np.ceil(sample_num/float(batch_size)))
        batch_index = np.arange(batch_num)

        if self.frequency == 'epoch':
            for epoch in range(epoch_num):
                state_dict_before_training = self.get_post.get_server_state_dict()
                model.load_state_dict()





class SynchronousWorker(object):
    '''
    distribute to spark worker by mapPartitions, works on spark worker
    '''
    def __init__():
        pass

    def train():
        pass
