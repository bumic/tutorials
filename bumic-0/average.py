'''
	TensorFlow Fold Tutorial
	average.py
	4.12.17

	Boston University Machine Intelligence Community
'''
from __future__ import division
import random
import tensorflow as tf
import tensorflow_fold as td
import numpy as np
import os
import matplotlib
import csv

matplotlib.use('Agg')
from matplotlib import pyplot as plt
sess = tf.InteractiveSession()
save_dir = "./save/model.ckpt"

def reduce_net_block():
    net_block = td.Concat() >> td.FC(20) >> td.FC(20) >> td.FC(1, activation=None) >> td.Function(lambda xs: tf.squeeze(xs, axis=1))
    return td.Map(td.Scalar()) >> td.Reduce(net_block)

def random_example():
    length = random.randrange(1, 10)
    data = [random.uniform(0,1) for _ in range(length)]
    result = sum(data)/len(data)
    return data, result

def train(batch_size=100):
	losses        = []
	net_block 	  = reduce_net_block()
	compiler 	  = td.Compiler.create((net_block, td.Scalar()))
	y, y_ 		  = compiler.output_tensors
	loss 		  = tf.nn.l2_loss(y - y_)
	train 		  = tf.train.AdamOptimizer().minimize(loss)
	saver    	  = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	validation_fd = compiler.build_feed_dict(random_example() for _ in range(1000))

	for i in range(10000):
		sess.run(train, compiler.build_feed_dict(random_example() for _ in range(batch_size)))
		loss_val = sess.run(loss, validation_fd)
		losses.append(loss_val)

		if i % 100 == 0:
			# print (i, loss_val)
			saver.save(sess, save_dir, global_step=i)
	
	fig = plt.figure()
	plt.plot(losses)
	plt.title('Training Loss')
	plt.xlabel('Batch')
	plt.ylabel('Loss')
	fig.savefig('./imgs/avg.png', dpi=fig.dpi)
	return net_block

print 'training...'
avg_block = train()
print 'testing...'
test_size = 100
correct_count = 0
correct   = []
incorrect = []
for i in range(test_size):
	sample, ans = random_example()
	hypo = avg_block.eval(sample)
	if abs(ans-hypo) < 0.01:
		correct.append((sample, hypo, ans))
		correct_count += 1
	else:
		incorrect.append((sample, hypo, ans))

with open("test-correct.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerows(correct)

with open("test-incorrect.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerows(incorrect)


