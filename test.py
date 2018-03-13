import optimizer
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

C = A(1,2)
C.bc(3)
print(C.a)

