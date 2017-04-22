import ctypes
from multiprocessing import Process, Array
import numpy
from multiprocessing import Queue


def f(a, q):
	a[0] = -a[0]
	q.put(numpy.asarray([1,3,5]))

if __name__ == '__main__':
	# Create the array
	N = int(10)
	unshared_arr = numpy.zeros(shape=(3,3))
	unshared_arr[0] = numpy.asarray([1, 2, 3])
	unshared_arr[1] = numpy.asarray([5, 6, 7])
	unshared_arr[2] = numpy.asarray([8, 9, 0])

	shape = unshared_arr.shape
	arr1d = unshared_arr.ravel()

	q = Queue()
	a = Array(ctypes.c_double, arr1d)
	print("Originally, the first two elements of arr = %s" % (a[:2]))

	# Create, start, and finish the child process
	p = Process(target=f, args=(a,q))
	p.start()
	p.join()

	# Print out the changed values
	print ("Now, the first two elements of arr = %s" % a[:2])

	while not q.empty():
		n = q.get()
		print(n.sum())


	# b = numpy.frombuffer(a.get_obj())
	# c = b.reshape(shape)
	# c[1][1] = 13
	# for d in c:
	# 	print(d[:])
	#
	#
	# # b[1] = 10.0
	# print(a[:])
