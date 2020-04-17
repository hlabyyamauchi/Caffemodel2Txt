import os
import numpy
import caffe
import csv

net = caffe.Net('./examples/mnist/lenet.prototxt', './examples/mnist/lenet.caffemodel', caffe.TEST);

for layername, params in net.params.items():
	print layername, params[0].data.shape, params[1].data.shape

layername = 'ip1'
isize = 500
jsize = 800

if os.path.isfile('./txt/' + layername + '_0.txt'):
	os.remove('./txt/' + layername + '_0.txt')
f = open('./txt/' + layername + '_0.txt', 'a')
print net.params[layername][0].data.shape
print net.params[layername][0].data
for i in xrange(isize):
	for j in xrange(jsize):
		#print i, ',', j, ':',net.params[layername][0].data[i, j]
		f.write(str(net.params[layername][0].data[i, j]) + '\n')
f.close()

if os.path.isfile('./txt/' + layername + '_1.txt'):
	os.remove('./txt/' + layername + '_1.txt')
f = open('./txt/' + layername + '_1.txt', 'a')
print net.params[layername][1].data.shape
print net.params[layername][1].data
for i in xrange(isize):
		f.write(str(net.params[layername][1].data[i]) + '\n')
f.close()



layername = 'ip2'
isize = 10
jsize = 500

if os.path.isfile('./txt/' + layername + '_0.txt'):
	os.remove('./txt/' + layername + '_0.txt')
f = open('./txt/' + layername + '_0.txt', 'a')
print net.params[layername][0].data.shape
print net.params[layername][0].data
for i in xrange(isize):
	for j in xrange(jsize):
		#print i, ',', j, ':',net.params[layername][0].data[i, j]
		f.write(str(net.params[layername][0].data[i, j]) + '\n')
f.close()

if os.path.isfile('./txt/' + layername + '_1.txt'):
	os.remove('./txt/' + layername + '_1.txt')
f = open('./txt/' + layername + '_1.txt', 'a')
print net.params[layername][1].data.shape
print net.params[layername][1].data
for i in xrange(isize):
		f.write(str(net.params[layername][1].data[i]) + '\n')
f.close()


layername = 'conv1'
isize = 20
jsize = 1
ksize = 5
lsize = 5

if os.path.isfile('./txt/' + layername + '_0.txt'):
	os.remove('./txt/' + layername + '_0.txt')
f = open('./txt/' + layername + '_0.txt', 'a')
print net.params[layername][0].data.shape
print net.params[layername][0].data
for i in xrange(isize):
	for j in xrange(jsize):
		for k in xrange(ksize):
			for l in xrange(lsize):
				#print i, ',', j, ':',net.params[layername][0].data[i, j]
				f.write(str(net.params[layername][0].data[i, j, k, l]) + '\n')
f.close()

if os.path.isfile('./txt/' + layername + '_1.txt'):
	os.remove('./txt/' + layername + '_1.txt')
f = open('./txt/' + layername + '_1.txt', 'a')
print net.params[layername][1].data.shape
print net.params[layername][1].data
for i in xrange(isize):
		f.write(str(net.params[layername][1].data[i]) + '\n')
f.close()


layername = 'conv2'
isize = 50
jsize = 20
ksize = 5
lsize = 5

if os.path.isfile('./txt/' + layername + '_0.txt'):
	os.remove('./txt/' + layername + '_0.txt')
f = open('./txt/' + layername + '_0.txt', 'a')
print net.params[layername][0].data.shape
print net.params[layername][0].data
for i in xrange(isize):
	for j in xrange(jsize):
		for k in xrange(ksize):
			for l in xrange(lsize):
				#print i, ',', j, ':',net.params[layername][0].data[i, j]
				f.write(str(net.params[layername][0].data[i, j, k, l]) + '\n')
f.close()

if os.path.isfile('./txt/' + layername + '_1.txt'):
	os.remove('./txt/' + layername + '_1.txt')
f = open('./txt/' + layername + '_1.txt', 'a')
print net.params[layername][1].data.shape
print net.params[layername][1].data
for i in xrange(isize):
		f.write(str(net.params[layername][1].data[i]) + '\n')
f.close()