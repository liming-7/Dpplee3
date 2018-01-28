'''
parameter server built with Flask
'''
from __future__ import absolute_import
import six.moves.cPickle as pickle
from flask import Flask, request
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

from rwlock import RWLock
#import optimizer

class ParamServer:
    ''' ParamServer usually works on master'''

    def __init__(self, network, mode, optimizer, lock):
        self.network = network
        self.weights = network.weights   #pytorch 获取model的weights
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
            self.pickled_weights = pickle.dumps(self.weights, -1)
            pickled_weights = self.pickled_weights
            if self.mode == 'asynchronous':
                self.lock.release()
            return pickled_weights

        @app.route('/update', methods=['POST'])
        def update_parameters():
            delta = pickle.loads(request.data)
            if self.mode == 'asynchronous':
                self.lock.acquire_write()

            if not self.network.built:
                self.network.build()

            constraints = self.network.model.constraints

            if len(constraints) == 0:
                def empty(a): return a
                constraints = [empty for x in self.weights]

            self.weights = self.optimizer.get_updates(self.weights, constraints, delta)

            if self.mode == 'asynchronous':
                self.lock.release()

            return 'Update done'

        self.app.run(host='0.0.0.0', debug=True,
                     threaded=True, use_reloader=False)
