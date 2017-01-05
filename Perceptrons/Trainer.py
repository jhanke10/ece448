from random import *

#Feature Variables
white = 0
black = 1

#Image Variables
width = 28
height = 28

def init_trainer():
	train_labels, train_images = parse_label_image('digitdata/trainingimages', 'digitdata/traininglabels')
	weights = init_weights(0)
	bias = init_bias(0)
	return train_labels, train_images, weights, bias

def parse_label_image(image, label):
	labels = []
	images = []

	image_file = open(image, 'r')
	label_file = open(label, 'r')

	#Get the training label values
	for line in label_file:
		labels.append(int(line))

	counter = 0
	#Get the image values
	image_map = []
	row = []
	for line in image_file:
		for char in line:
			if char == ' ':
				row.append(white)
			elif char == '+':
				row.append(black)
			elif char == '#':
				row.append(black)
		image_map.append(row)
		row = []
		counter += 1
		if counter == height:
			counter = 0
			images.append(image_map)
			image_map = []
	return labels, images

def create_trainer(epochs, use_bias, random):
	train_labels, train_images, weights, bias = init_trainer()
	accurate = 0
	total = 0

	if not random:
		for image in range(len(train_images)):
			label = train_labels[image]

			#Calculate scores for different labels
			scores = []
			for labels in range(10):
				score = 0
				for i in range(height):
					for j in range(width):
						score += weights[labels][i][j] * train_images[image][i][j]
				if use_bias:
					score += bias[labels]
				scores.append(score)

			#Decision for label of training image
			label_image = scores.index(max(scores))

			if label == label_image:
				accurate += 1
			else:
				for i in range(0, height):
					for j in range(0, width):
						weights[label][i][j] += alpha(epochs) * train_images[image][i][j]
						weights[label_image][i][j] -= alpha(epochs) * train_images[image][i][j]
				if use_bias:
					bias[labels] += alpha(epochs)
					bias[label_image] -= alpha(epochs)

			total += 1

	elif random:
		read = []
		while len(read) < len(train_images):
			image = randint(0, len(train_images)-1)
			while image in read:
				image = randint(0, len(train_images)-1)
			read.append(image)

			label = train_labels[image]

			#Calculate scores for different labels
			scores = []
			for labels in range(10):
				score = 0
				for i in range(height):
					for j in range(width):
						score += weights[labels][i][j] * train_images[image][i][j]
				if use_bias:
					score += bias[labels]
				scores.append(score)

			#Decision for label of training image
			label_image = scores.index(max(scores))

			if label == label_image:
				accurate += 1
			else:
				for i in range(0, height):
					for j in range(0, width):
						weights[label][i][j] += alpha(epochs) * train_images[image][i][j]
						weights[label_image][i][j] -= alpha(epochs) * train_images[image][i][j]
				if use_bias:
					bias[labels] += alpha(epochs)
					bias[label_image] -= alpha(epochs)

			total += 1


	return (float(accurate) / total) * 100, weights, bias


def init_bias(bias_num):
	bias = []
	for i in range(10):
		bias.append(bias_num)
	return bias

def init_weights(weight):
	weights = []
	for c in range(10):
		class_weight = []
		for i in range(height):
			row = []
			for j in range(width):
				row.append(randint(0,weight))
			class_weight.append(row)
		weights.append(class_weight)
	return weights

def alpha(epochs):
	return float(1000) / (1000 + epochs)