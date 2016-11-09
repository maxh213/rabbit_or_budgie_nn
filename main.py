import tensorflow as tf

RABBIT = [0]
BUDGIE = [1]

def get_images(filename, amount):
	filename_queue = tf.train.string_input_producer(
	tf.train.match_filenames_once("./" + filename + "/*.jpg"))

	image_reader = tf.WholeFileReader()

	images = []
	for i in range(amount):
		_, image_file = image_reader.read(filename_queue)
		image = tf.image.decode_jpeg(image_file)
		image.set_shape((28, 28, 3))
		images.append(image)
	return images

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[280, 280, 3])
y_ = tf.placeholder(tf.float32, shape=[1])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.squeeze(y_conv)
#y_ = tf.pad(y_, tf.shape(y_conv), mode='CONSTANT')
y_conv = tf.gather(y_conv, [1])
#y_conv = tf.sigmoid(y_conv)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


TRAINING_IMAGE_AMOUNT = 1000
TESTING_IMAGE_AMOUNT = 50

rabbit_images = get_images("rabbits", TRAINING_IMAGE_AMOUNT)
test_rabbit_images = rabbit_images[TESTING_IMAGE_AMOUNT:]
rabbit_images = rabbit_images[:TRAINING_IMAGE_AMOUNT-TESTING_IMAGE_AMOUNT]

budgie_images = get_images("budgies", TRAINING_IMAGE_AMOUNT)
test_budgie_images = budgie_images[TESTING_IMAGE_AMOUNT:]
budgie_images = budgie_images[:TRAINING_IMAGE_AMOUNT-TESTING_IMAGE_AMOUNT]

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

switch = False
#*2 because rabbits and budgie images 
for i in range((TRAINING_IMAGE_AMOUNT * 2) - (TESTING_IMAGE_AMOUNT * 2)):
	try: 
		if switch:
			output = sess.run(y_conv, feed_dict={x:budgie_images[i].eval(), y_: BUDGIE, keep_prob: 1.0})
			print("STEP " + str(i) + ": Correct output: " + str(BUDGIE[0]) + " got: " + str(output))
			entropy, _ = sess.run([cross_entropy,train_step], feed_dict={x:budgie_images[i].eval(), y_: BUDGIE, keep_prob: 0.5})
			print("ENTROPY: " + str(entropy))
		else:
			output = sess.run(y_conv, feed_dict={x:rabbit_images[i].eval(), y_: RABBIT, keep_prob: 1.0})
			print("STEP " + str(i) + ": Correct output: " + str(RABBIT[0]) + " got: " + str(output))
			entropy, _ = sess.run([cross_entropy,train_step],feed_dict={x:rabbit_images[i].eval(), y_: RABBIT, keep_prob: 0.5})
			print("ENTROPY: " + str(entropy))
		if i % 10 == 0:
			if switch:
				switch = False
			else:
				switch = True
	except ValueError: 
		print("ERROR")

#accuracy calculation
correct = 0
for i in range(TESTING_IMAGE_AMOUNT * 2):
	if switch:
		output = sess.run(y_conv, feed_dict={x:budgie_images[i].eval(), y_: BUDGIE, keep_prob: 1.0})
		print("TESTING STEP " + str(i) + ": Correct output: " + str(BUDGIE[0]) + " got: " + str(output))
		if output[0] >= 0.5:
			correct += 1
	else:
		output = sess.run(y_conv, feed_dict={x:rabbit_images[i].eval(), y_: RABBIT, keep_prob: 1.0})
		print("TESTING STEP " + str(i) + ": Correct output: " + str(RABBIT[0]) + " got: " + str(output))
		if output[0] < 0.5:
			correct += 1
	if i % 5 == 0:
		if switch:
			switch = False
		else:
			switch = True	

print ("Correct: " + str(correct) + "/" + str(TESTING_IMAGE_AMOUNT * 2))
	
coord.request_stop()
coord.join(threads)
