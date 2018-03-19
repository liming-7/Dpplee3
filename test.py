import optimizer
# import six.moves.cPickle as pickle
import pickle
import json
class A:
	def __init__(self,a,b):
		self.a = a
		self.b = b

	def ac(self, c):
		self.a = c
		return c

	def bc (self, d):
		self.ac(d)
		return 1

a=pickle.dumps(A)
print(type(a))
b=pickle.loads(a)
print(b)
print(type(b))

# j=json.dumps(a)
# print(type(j))
# ja=json.loads(j)
# print(type(ja))
