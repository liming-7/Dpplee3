'''
get or post server weights
'''
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

def get_server_weights(master_url='localhost:5000'):
    '''
    get weights from parameter server
    '''
    request = urllib2.Request('http://{0}/parameters'.format(master_url))
    request.add_header('Content-Type', 'application/dpplee3')
    data = urllib2.urlopen(request).read()
    weights = pickle.loads(data)
    return weights

def post_delta_to_server(delta, master_url='localhost:5000'):
    '''
    post delta to parameter server and waiting for update
    '''
    request = urllib2.Request('http://{0}/update'.format(master_url))
    request.add_header('Content-Type', 'application/dpplee3')
    data = pickle.dumps(delta, -1)
    return urllib2.urlopen(request, data).read()
