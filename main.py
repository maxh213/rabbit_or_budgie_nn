import tensorflow as tf

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

rabbit_images = get_images("rabbits", 100)
budgie_images = get_images("budgies", 100)

with tf.Session() as sess:
	# Required to get the filename matching to run.
	tf.initialize_all_variables().run()

	# Coordinate the loading of image files.
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	# Get an image tensor and print its value.
	print(sess.run([rabbit_images[0]]))
	print(sess.run([budgie_images[0]]))

	# Finish off the filename queue coordinator.
	coord.request_stop()
	coord.join(threads)


