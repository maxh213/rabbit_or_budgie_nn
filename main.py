import tensorflow as tf

def get_output(output):
	array = []
	for i in range(100):
		array.append([])
		for j in range(100):
			if j == output:
				array[i].append(999)
			else:
				array[i].append(output)
	return array

RABBIT = get_output(0)
BUDGIE = get_output(1)

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
y_ = tf.placeholder(tf.float32, shape=[100, 100])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,280,280,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 5024])
b_fc1 = bias_variable([5024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([5024, 100])
b_fc2 = bias_variable([100])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

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

budgie_incorrect_count = 0
rabbit_incorrect_count = 0
budgie_switch = False
rabbit_switch = False
budgie_error_run = 2
rabbit_error_run = 2
current_error_run = 0

#*2 because rabbits and budgie images 
for i in range((TRAINING_IMAGE_AMOUNT * 2) - (TESTING_IMAGE_AMOUNT * 2)):
	try: 
		if switch:
			entropy, _ = sess.run([cross_entropy,train_step], feed_dict={x:budgie_images[i].eval(), y_: BUDGIE, keep_prob: 0.5})
			print("STEP" + str(i) + ": BUDGIE ENTROPY: " + str(entropy))
			output = sess.run(y_conv, feed_dict={x:budgie_images[i].eval(), y_: BUDGIE, keep_prob: 1.0})
			correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(BUDGIE,1))
			#tf.nn.in_top_k(output, BUDGIE, 1)#tf.equal(tf.argmax(output,1), tf.argmax(BUDGIE,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			#print(output)
			#print(BUDGIE)
			print(sess.run(tf.argmax(output,1)))
			print(sess.run(tf.argmax(BUDGIE,1)))
			print(sess.run(accuracy))
			if sess.run(accuracy) == 0.0:
				budgie_incorrect_count += 1
			else: 
				budgie_incorrect_count = 0
			print("---")
		else:
			entropy, _ = sess.run([cross_entropy,train_step],feed_dict={x:rabbit_images[i].eval(), y_: RABBIT, keep_prob: 0.75})
			print("STEP" + str(i) + ": RABBIT ENTROPY: " + str(entropy))
			output = sess.run(y_conv, feed_dict={x:rabbit_images[i].eval(), y_: RABBIT, keep_prob: 1.0})
			correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(RABBIT,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			#print(output)
			#print(RABBIT)
			print(sess.run(tf.argmax(output,1)))
			print(sess.run(tf.argmax(RABBIT,1)))
			print(sess.run(accuracy))
			if sess.run(accuracy) == 0.0:
				rabbit_incorrect_count += 1
			else:
				rabbit_incorrect_count = 0

		#print("Rabbit incorrect in a row count: " + str(rabbit_incorrect_count))
		#print("Budgie incorrect in a row count: " + str(budgie_incorrect_count))
		#if rabbit_incorrect_count > rabbit_error_run:
		#	rabbit_switch = True
		#if budgie_incorrect_count > budgie_error_run:
		#	budgie_switch = True

		if budgie_switch or rabbit_switch:
			current_error_run += 1
			print("CURRENT ERROR RUN: " + str(current_error_run))
		if i % 1 == 0:
			if switch == False:
				switch = True
			else:
				switch = False
		if budgie_switch:
			print("ON BUDGIE ERROR RUN")
			switch = True
			i += 1
		if rabbit_switch:
			print("ON RABBIT ERROR RUN")
			switch = False
			i += 1
		if current_error_run % budgie_error_run == 0:
			if budgie_switch:
				budgie_incorrect_count = 0
			if rabbit_switch:
				rabbit_incorrect_count = 0
			budgie_switch = False
			rabbit_switch = False
			current_error_run = 0




	except ValueError: 
		print("ERROR")


#accuracy calculation
correct = 0
for i in range(TESTING_IMAGE_AMOUNT * 2):
	if switch:
		output = sess.run(y_conv, feed_dict={x:budgie_images[i].eval(), y_: BUDGIE, keep_prob: 1.0})
		correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(BUDGIE,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		if accuracy < 0.5:
			correct += 1
	else:
		output = sess.run(y_conv, feed_dict={x:rabbit_images[i].eval(), y_: RABBIT, keep_prob: 1.0})
		correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(RABBIT,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		if accuracy < 0.5:
			correct += 1
	if i % 5 == 0:
		if switch:
			switch = False
		else:
			switch = True	

print ("Testing results: " + str(correct) + "/" + str(TESTING_IMAGE_AMOUNT * 2))
	
coord.request_stop()
coord.join(threads)
