from Trainer import *

def test_trainer(label, image, weight, bias, use_bias):
	total = 0
	accurate = 0
	confusion_matrix = []

	#Confusion matrix
	for i in range(10):
		row = []
		for j in range(10):
			row.append(0)
		confusion_matrix.append(row)

	test_labels, test_images = parse_label_image(image, label)

	for images in range(len(test_images)):
		real_label = test_labels[images]

		#Calculate scores for different labels
		scores = []
		for labels in range(10):
			score = 0
			for i in range(height):
				for j in range(width):
					score += weight[labels][i][j] * test_images[images][i][j]
			if use_bias:
				score += bias[labels]
			scores.append(score)

		#Decision for label of test image
		label_image = scores.index(max(scores))

		if real_label == label_image:
			accurate += 1

		confusion_matrix[real_label][label_image] += 1
		total += 1

	return (float(accurate) / total) * 100, confusion_matrix, total

def main():
	weights = []
	bias = []
	epochs = 0
	accuracy = 0

	print 'Training...'
	while epochs < 30:
		accuracy, weights, bias = create_trainer(epochs, False, False)
		print 'Percent Accurate (epoch = ' + str(epochs) + '): ' + str(accuracy) + '%'
		epochs += 1

	print ''
	print 'Testing...'
	accuracy, confusion_matrix, total_docs = test_trainer('digitdata/testlabels', 'digitdata/testimages', weights, bias, False)
	print 'Percent Accurate: ' + str(accuracy) + '%'

	print ''
	print 'Confusion Matrix'
	print 'Total Images = ' + str(total_docs)
	print 'Predicted (x-axis: 0-9) Actual (y-axis: 0-9)'
	for i in range(10):
		row = ""
		for j in range(10):
			row += str(confusion_matrix[i][j]) + "     "
		print row

if __name__ == "__main__":main()