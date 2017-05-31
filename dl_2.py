#coding: utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.computational_graph as cg
import chainer.datasets.tuple_dataset as td
import chainer.links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions
import time, csv
import numpy as np

def unpickle(f):
	import pickle
	fo = open(f, 'rb')
	d = pickle.load(fo)
	fo.close()
	return d

def load_cifar10(datadir):
        x_train = None
        y_train = []

        x_train = np.loadtxt('testdata.csv', delimiter=',',dtype='int')

	y_train = [1]


	x_train = x_train.reshape(len(x_train)/32,1,32,32)
        x_test = x_train
	y_test = [1]

        print(x_train)
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)
	x_train /= 255
	x_test /= 255
	y_train = np.array(y_train)
	y_test = np.array(y_test)

        print(y_train)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        print(len(x_train))
	train = td.TupleDataset(x_train, y_train)
	test = td.TupleDataset(x_test, y_test)
        return train, test

class Cifar10Model(chainer.Chain):
	def __init__(self):
		super(Cifar10Model,self).__init__(
			conv1 = F.Convolution2D(1, 32, 1, pad=1),
			conv2 = F.Convolution2D(32, 32, 1, pad=1),
			conv3 = F.Convolution2D(32, 32, 1, pad=1),
			conv4 = F.Convolution2D(32, 32, 1, pad=1),
			conv5 = F.Convolution2D(32, 32, 1, pad=1),
			conv6 = F.Convolution2D(32, 32, 1, pad=1),
			l1 = L.Linear(2048, 512),
			l2 = L.Linear(512,1))

	def __call__(self, x, train=True):
		h = F.relu(self.conv1(x))
		h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
		h = F.relu(self.conv3(h))
		h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
		h = F.relu(self.conv5(h))
		h = F.max_pooling_2d(F.relu(self.conv6(h)), 2)
		h = F.dropout(F.relu(self.l1(h)), train=train)
		return self.l2(h)

if __name__ == "__main__":
	gpu_flag = 0

	if gpu_flag >= 0:
		cuda.check_cuda_available()
	xp = cuda.cupy if gpu_flag >= 0 else np

	batchsize = 100
	n_epoch = 20

	print("load CIFAR-10 dataset")

	train, test = load_cifar10("/home/senami/デスクトップ/cifar-10-batches-py")

	model = L.Classifier(Cifar10Model())
	if gpu_flag >= 0:
		chainer.cuda.get_device(gpu_flag).use() # Make a specified GPU current
		model.to_gpu() # Copy the model to the GPU

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	train_iter = chainer.iterators.SerialIterator(train, 100)
	test_iter = chainer.iterators.SerialIterator(test, 100,repeat=False, shuffle=False)

	updater = training.StandardUpdater(train_iter, optimizer, device=gpu_flag)
	trainer = training.Trainer(updater, (40, 'epoch'), out='result')
	trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_flag))

	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
	trainer.extend(extensions.ProgressBar())
	trainer.run()

	model.to_cpu()

#chainer.serializers.save_hdf5("cipar10.model", model);
