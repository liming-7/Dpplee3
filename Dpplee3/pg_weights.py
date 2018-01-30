'''
get or post server weights
'''
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

class Get_post_state_dict(object):

    def __init__(master_url, sparkAppname = None)
    '''
    master_url is master_url at most time, to reduce the net costs
    sparkAppname is for after
    '''
        self.url = master_url
        self.sparkAppname = sparkAppname ##用于以后功能

    def get_server_state_dict(url=self.url):
        '''
        get weights from parameter server
        '''
        request = urllib2.Request('http://{0}/parameters'.format(url))
        request.add_header('Content-Type', 'application/dpplee3')
        data = urllib2.urlopen(request).read()
        weights = pickle.loads(data)
        return state_dict

    def post_delta_to_server(delta, url=self.url):
        '''
        post delta to parameter server and waiting for update
        '''
        request = urllib2.Request('http://{0}/worker_updates'.format(url))
        request.add_header('Content-Type', 'application/dpplee3')
        data = pickle.dumps(delta, -1)
        return urllib2.urlopen(request, data).read()
