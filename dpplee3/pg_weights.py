'''
get or post server weights
'''
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import six.moves.cPickle as pickle


class Get_post_state_dict(object):

    def __init__(self, master_url, sparkAppname = None):
        """
        master_url is master_url at most time, to reduce the net costs
        sparkAppname is for after
        """
        self.url = master_url
        self.sparkAppname = sparkAppname
        '''for after usage'''

    def get_server_state_dict(self):
        '''
        get weights from parameter server
        '''
        request = urllib2.Request('http://{0}/parameters'.format(self.url))
        request.add_header('Content-Type', 'application/dpplee3')
        data = urllib2.urlopen(request).read()
        state_dict = pickle.loads(data)
        return state_dict

    def post_updates_to_server(self, delta):
        '''
        post delta to parameter server and waiting for update
        '''
        request = urllib2.Request('http://{0}/worker_updates'.format(self.url))
        request.add_header('Content-Type', 'application/dpplee3')
        data = pickle.dumps(delta, -1)
        return urllib2.urlopen(request, data).read()
