import numpy as np
from math import sqrt

# Attention problem avec quartile a voir

class TinyStatistician:
	def __init__(self):
		pass

	def mean(self, x):
		m = len(x)
		b = 0
		for el in x:
			b += el
		return float(b / m)

	def median(self, x):
		if self.__check_input(x) == False:
			return None
		# Demi some ou pas ????
		b = list(x)
		b.sort()
		if len(b) % 2 != 0:
			return float(b[len(b) // 2])
		else:
			return float((b[len(b) // 2] + b[len(b) // 2 - 1]) / 2)

	def quartile(self, x):
		if self.__check_input(x) == False:
			return None
		b = list(x)
		b.sort()
		r = []
		# print(len(b) * 0.25)
		# print(len(b) * 0.75)
		r.append(float(b[int(len(b) * 0.25)]))
		r.append(float(b[int(len(b) * 0.75)]))
		return r

	def percentile(self, x, p):
		#
		if self.__check_input(x) == False:
			return None
		if p < 0 or p > 100:
			return None
		b = list(x)
		b.sort()
		return float(b[int(len(b) * (p / 100))])

	def var(self, x):
		m = len(x)
		mean = self.mean(x)
		sum = 0
		for i in range(0, m):
			sum += ((x[i] - mean) * (x[i] - mean))
		return float(sum / m)

	def std(self, x):
		return float(sqrt(self.var(x)))

	def __check_input(self, x):
		try:
			if isinstance(x, list) == False:
				# print(type(x).__module__)
				if type(x).__module__ != np.__name__:
					raise TypeError("Error: Input must be a np.array or a list")
			if len(x) == 0:
				raise ValueError("Error: Input array cannot be empty (div by 0)")
			for el in x:
				if isinstance(el, (float, int)) == False and isinstance(el.item(), (float, int)) == False:
					raise ValueError("Error: All array elements must be float or int")
			return True
		except:
			return False
