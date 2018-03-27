'''
parameter server built with Flask
'''
from __future__ import absolute_import
import six.moves.cPickle as pickle
from multiprocessing import Process
from flask import Flask, request
try:
    import urllib.request as  urllib2
except ImportError:
    import urllib2

from Dpplee3.rwlock import RWLock
from Dpplee3 import optimizer
#import optimizer

class ParamServer(object):
    ''' ParamServer usually works on master'''

    def __init__(self, network, mode, optimizer, lock):
        self.network = network
        self.state_dict = network.state_dict   #pytorch 获取model的weights
        self.mode = mode
        self.optimizer = optimizer
        self.lock = lock

    def start_server(self):
        ''' Start parameter server'''
        self.server = Process(target=self.service)
        self.server.start()

    def stop_server(self):
        ''' Terminate parameter server'''
        self.server.terminate()
        self.server.join()

    def service(self):
        ''' Define service and run flask app'''
        app = Flask(__name__)
        self.app = app

        @app.route('/')
        def home():
            return 'Dpplee3'

        @app.route('/parameters', methods=['GET'])
        def get_parameters():
            if self.mode == 'asynchronous':
                self.lock.acquire_read()
            self.pickled_weights = pickle.dumps(self.state_dict, -1)
            pickled_weights = self.pickled_weights
            if self.mode == 'asynchronous':
                self.lock.release()
            return pickled_weights

        @app.route('/update', methods=['POST'])
        def update_parameters():
            delta = pickle.loads(request.data) #get the update infomation from workers
            if self.mode == 'asynchronous':
                self.lock.acquire_write()

            self.network.train() #set the training or not training, is useful for batchnorm and dropout
            self.optimizer = optimizer.SGD(self.network.parameters(), lr=0.001, momentum=0.5)
            optimizer.zero_grad()
            self.optimizer.replace_grad(delta)
            self.optimizer.step()

            if self.mode == 'asynchronous':
                self.lock.release()

            return 'Update done'

        self.app.run(host='0.0.0.0', debug=True,
                     threaded=True, use_reloader=False)
