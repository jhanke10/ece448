from math import log
import operator

#Local Variables
num_pos_words = 0
num_neg_words = 0
num_unique_words = 0
num_pos_doc = 0
num_neg_doc = 0
total_doc = 0

def create_multinomial_trainer(file_path):
	#Create Dictionaries
	trainer = {}
	pos_words = {}
	neg_words = {}

	#Create Global Variables
	global num_pos_words
	global num_neg_words
	global num_unique_words
	global num_pos_doc
	global num_neg_doc
	global total_doc

	#Read file and create conditional probabilities
	corpus = open(file_path, 'r')
	for line in corpus:
		line = line.split()
		label = int(line[0])
		line.remove(line[0])

		#Add words to dictionaries
		for pair in line:
			word, count = pair.split(':')
			if label > 0:
				if pos_words.has_key(word):
					pos_words[word] += int(count)
				else:
					pos_words[word] = int(count)
				num_pos_words += int(count)
			else:
				if neg_words.has_key(word):
					neg_words[word] += int(count)
				else:
					neg_words[word] = int(count)
				num_neg_words += int(count)

		#Calculate number of documents
		if label > 0:
			num_pos_doc += 1
		else:
			num_neg_doc += 1

	#Add words that are not found in other set
	for words in pos_words:
		if words not in neg_words:
			neg_words[words] = 0
	for words in neg_words:
		if words not in pos_words:
			pos_words[words] = 0

	#Calculate number of unique words
	unique = set()
	for key in pos_words:
		unique.add(key)
	for keys in neg_words:
		unique.add(keys)
	num_unique_words = len(unique)

	#Calculate conditional probability
	for words in pos_words:
		pos_words[words] = float(pos_words[words] + 1) / (num_pos_words + num_unique_words)
	for words in neg_words:
		neg_words[words] = float(neg_words[words] + 1) / (num_neg_words + num_unique_words)
	
	#Add words to dictionary
	trainer[1] = pos_words
	trainer[-1] = neg_words
	total_doc = num_pos_doc + num_neg_doc

	#Caluculate odds P(word, 1)/P(word, -1)
	odds_word = {}
	for words in unique:
		odds_word[words] = float(pos_words[words]) / neg_words[words]

	#Sort lists by percentage
	sort_pos = sorted(pos_words.items(), key = operator.itemgetter(1), reverse = True)
	sort_neg = sorted(neg_words.items(), key = operator.itemgetter(1), reverse = True)
	sort_odds = sorted(odds_word.items(), key = operator.itemgetter(1), reverse = True)

	return (trainer, sort_pos, sort_neg, sort_odds)

def multinomial_bayes(trainer, file_path):
	#Variables to calculate accuracy
	num_docs = 0
	accurate_docs = 0

	#Variables for confusion matrix
	num_docs_pos2 = 0
	num_docs_neg2 = 0
	num_docs_pos_neg = 0
	num_docs_neg_pos = 0

	#Read file and create conditional probabilities
	corpus = open(file_path, 'r')
	for line in corpus:
		line = line.split()
		label = int(line[0])
		line.remove(line[0])

		#Probability variables
		pos_prob = log(float(num_pos_doc) / total_doc)
		neg_prob = log(float(num_neg_doc) / total_doc)

		#Calculate probabilites
		for pair in line:
			word, count = pair.split(':')
			if trainer[1].has_key(word):
				pos_prob += log(trainer[1][word]) * int(count)
			if trainer[-1].has_key(word):
				neg_prob += log(trainer[-1][word]) * int(count)

		#Determine guess
		outcome = 0
		if pos_prob > neg_prob:
			outcome = 1
		else:
			outcome = -1

		#Determine accurate
		if label == outcome:
			accurate_docs += 1
			if label == 1:
				num_docs_pos2 += 1
			else:
				num_docs_neg2 += 1
		else:
			if label == 1:
				num_docs_pos_neg += 1
			else:
				num_docs_neg_pos += 1
		num_docs += 1

	return ((float(accurate_docs) / num_docs) * 100, num_docs, num_docs_pos2, num_docs_pos_neg, num_docs_neg_pos, num_docs_neg2)

def create_bernoulli_trainer(file_path):
	#Create Dictionaries
	trainer = {}
	pos_words = {}
	neg_words = {}

	#Create Global Variables
	global num_pos_doc
	global num_neg_doc
	global total_doc

	#Read file and create conditional probabilities
	corpus = open(file_path, 'r')
	for line in corpus:
		line = line.split()
		label = int(line[0])
		line.remove(line[0])

		#Add words to dictionaries
		for pair in line:
			word, count = pair.split(':')
			if label > 0:
				if pos_words.has_key(word):
					pos_words[word] += 1
				else:
					pos_words[word] = 1
			else:
				if neg_words.has_key(word):
					neg_words[word] += 1
				else:
					neg_words[word] = 1

		#Calculate number of documents
		if label > 0:
			num_pos_doc += 1
		else:
			num_neg_doc += 1

	total_doc = num_pos_doc + num_neg_doc

	#Add words that are not found in other set
	for words in pos_words:
		if words not in neg_words:
			neg_words[words] = 0
	for words in neg_words:
		if words not in pos_words:
			pos_words[words] = 0

	#Calculate number of unique words
	unique = set()
	for key in pos_words:
		unique.add(key)
	for keys in neg_words:
		unique.add(keys)
	num_unique_words = len(unique)

	#Calculate conditional probability
	for words in pos_words:
		pos_words[words] = float(pos_words[words] + 1) / (num_pos_doc + total_doc)
	for words in neg_words:
		neg_words[words] = float(neg_words[words] + 1) / (num_neg_doc + total_doc)

	#Add words to dictionary
	trainer[1] = pos_words
	trainer[-1] = neg_words
	total_doc = num_pos_doc + num_neg_doc

	#Caluculate odds P(word, 1)/P(word, -1)
	odds_word = {}
	for words in unique:
		odds_word[words] = float(pos_words[words]) / neg_words[words]

	#Sort lists by percentage
	sort_pos = sorted(pos_words.items(), key = operator.itemgetter(1), reverse = True)
	sort_neg = sorted(neg_words.items(), key = operator.itemgetter(1), reverse = True)
	sort_odds = sorted(odds_word.items(), key = operator.itemgetter(1), reverse = True)

	return (trainer, sort_pos, sort_neg, sort_odds)

def bernoulli_bayes(trainer, file_path):
	#Variables to calculate accuracy
	num_docs = 0
	accurate_docs = 0

	#Variables for confusion matrix
	num_docs_pos2 = 0
	num_docs_neg2 = 0
	num_docs_pos_neg = 0
	num_docs_neg_pos = 0

	#Variable for words in doc
	doc_word = []

	#Probabilities where (X = 0)
	not_pos_prob = 0
	not_neg_prob = 0

	#Calculate prob of not in training 
	for words in trainer[1]:
		not_pos_prob += log(1 - trainer[1][words])
	for words in trainer[-1]:
		not_neg_prob += log(1 - trainer[-1][words])

	#Read file and create conditional probabilities
	corpus = open(file_path, 'r')
	for line in corpus:
		line = line.split()
		label = int(line[0])
		line.remove(line[0])

		#Probability variables
		pos_prob = log(float(num_pos_doc) / total_doc)
		neg_prob = log(float(num_neg_doc) / total_doc)
		not_pos_prob_doc = not_pos_prob
		not_neg_prob_doc = not_neg_prob

		#Calculate probabilites
		for pair in line:
			word, count = pair.split(':')

			#Words that have appeared (X = 1)
			if trainer[1].has_key(word):
				pos_prob += log(trainer[1][word])
			if trainer[-1].has_key(word):
				neg_prob += log(trainer[-1][word])
			doc_word.append(word)

		#Words that have not appeared (X = 0)
		for words in doc_word:
			if words in trainer[1]:
				not_pos_prob_doc -= log(1 - trainer[1][words]) 
		for words in doc_word:
			if words in trainer[-1]:
				not_neg_prob_doc -= log(1 - trainer[-1][words])
		pos_prob += not_pos_prob_doc
		neg_prob += not_neg_prob_doc

		#Determine guess
		outcome = 0
		if pos_prob > neg_prob:
			outcome = 1
		else:
			outcome = -1

		#Determine accurate
		if label == outcome:
			accurate_docs += 1
			if label == 1:
				num_docs_pos2 += 1
			else:
				num_docs_neg2 += 1
		else:
			if label == 1:
				num_docs_pos_neg += 1
			else:
				num_docs_neg_pos += 1
		num_docs += 1

		doc_word = []

	return ((float(accurate_docs) / num_docs) * 100, num_docs, num_docs_pos2, num_docs_pos_neg, num_docs_neg_pos, num_docs_neg2)

def main():
	trainer, pos_list, neg_list, odds_list = create_multinomial_trainer('movie_review/rt-train.txt')
	accurate, num_docs, pos2, pos_neg, neg_pos, neg2 = multinomial_bayes(trainer, 'movie_review/rt-test.txt')

	print 'Percent Accurate Multinomial Movie: '  + str(accurate) + '%'

	print 'Number of 1 Docs: ' + str(num_pos_doc) + " Classification Rate: " + str((float(num_pos_doc)/total_doc * 100)) + "%"
	print 'Number of -1 Docs: ' + str(num_neg_doc) + " Classification Rate: " + str((float(num_neg_doc)/total_doc * 100)) + "%"
	print ''

	print 'Confusion Matrix:'
	print "n = " + str(num_docs) + "    Predicted: 1    Predicted: -1"
	print "Actual:  1       " + str(pos2)    + "               " + str(pos_neg)
	print "Actual: -1       " + str(neg_pos) + "               " + str(neg2)
	print ""

	print "Top 10 Likely for Label = 1: "
	for i in range(1, 11):
		print str(i) + ": " + pos_list[i-1][0]
	print ""

	print "Top 10 Likely for Label = -1: "
	for i in range(1, 11):
		print str(i) + ": " + neg_list[i-1][0]
	print ""

	print "Top 10 Odds Ratio: "
	for i in range(1, 11):
		print str(i) + ": " + odds_list[i-1][0]
	print ""


	trainer, pos_list, neg_list, odds_list = create_bernoulli_trainer('movie_review/rt-train.txt')
	accurate, num_docs, pos2, pos_neg, neg_pos, neg2 = bernoulli_bayes(trainer, 'movie_review/rt-test.txt')

	print 'Percent Accurate Bernoulli Movie: '  + str(accurate) + '%'

	print 'Number of 1 Docs: ' + str(num_pos_doc) + " Classification Rate: " + str((float(num_pos_doc)/total_doc * 100)) + "%"
	print 'Number of -1 Docs: ' + str(num_neg_doc) + " Classification Rate: " + str((float(num_neg_doc)/total_doc * 100)) + "%"
	print ''

	print 'Confusion Matrix:'
	print "n = " + str(num_docs) + "    Predicted: 1    Predicted: -1"
	print "Actual:  1       " + str(pos2)    + "               " + str(pos_neg)
	print "Actual: -1       " + str(neg_pos) + "               " + str(neg2)
	print ""

	print "Top 10 Likely for Label = 1: "
	for i in range(1, 11):
		print str(i) + ": " + pos_list[i-1][0]
	print ""

	print "Top 10 Likely for Label = -1: "
	for i in range(1, 11):
		print str(i) + ": " + neg_list[i-1][0]
	print ""

	print "Top 10 Odds Ratio: "
	for i in range(1, 11):
		print str(i) + ": " + odds_list[i-1][0]
	print ""

	trainer, pos_list, neg_list, odds_list = create_multinomial_trainer('fisher_2topic/fisher_train_2topic.txt')
	accurate, num_docs, pos2, pos_neg, neg_pos, neg2 = multinomial_bayes(trainer, 'fisher_2topic/fisher_test_2topic.txt')

	print 'Percent Accurate Multinomial Topic: '  + str(accurate) + '%'

	print 'Number of 1 Docs: ' + str(num_pos_doc) + " Classification Rate: " + str((float(num_pos_doc)/total_doc * 100)) + "%"
	print 'Number of -1 Docs: ' + str(num_neg_doc) + " Classification Rate: " + str((float(num_neg_doc)/total_doc * 100)) + "%"
	print ''

	print 'Confusion Matrix:'
	print "n = " + str(num_docs) + "    Predicted: 1    Predicted: -1"
	print "Actual:  1       " + str(pos2)    + "               " + str(pos_neg)
	print "Actual: -1       " + str(neg_pos) + "               " + str(neg2)
	print ""

	print "Top 10 Likely for Label = 1: "
	for i in range(1, 11):
		print str(i) + ": " + pos_list[i-1][0]
	print ""

	print "Top 10 Likely for Label = -1: "
	for i in range(1, 11):
		print str(i) + ": " + neg_list[i-1][0]
	print ""

	print "Top 10 Odds Ratio: "
	for i in range(1, 11):
		print str(i) + ": " + odds_list[i-1][0]
	print ""

	trainer, pos_list, neg_list, odds_list = create_bernoulli_trainer('fisher_2topic/fisher_train_2topic.txt')
	accurate, num_docs, pos2, pos_neg, neg_pos, neg2 = bernoulli_bayes(trainer, 'fisher_2topic/fisher_test_2topic.txt')

	print 'Percent Accurate Bernoulli Topic: '  + str(accurate) + '%'

	print 'Number of 1 Docs: ' + str(num_pos_doc) + " Classification Rate: " + str((float(num_pos_doc)/total_doc * 100)) + "%"
	print 'Number of -1 Docs: ' + str(num_neg_doc) + " Classification Rate: " + str((float(num_neg_doc)/total_doc * 100)) + "%"
	print ''

	print 'Confusion Matrix:'
	print "n = " + str(num_docs) + "    Predicted: 1    Predicted: -1"
	print "Actual:  1       " + str(pos2)    + "               " + str(pos_neg)
	print "Actual: -1       " + str(neg_pos) + "               " + str(neg2)
	print ""

	print "Top 10 Likely for Label = 1: "
	for i in range(1, 11):
		print str(i) + ": " + pos_list[i-1][0]
	print ""

	print "Top 10 Likely for Label = -1: "
	for i in range(1, 11):
		print str(i) + ": " + neg_list[i-1][0]
	print ""

	print "Top 10 Odds Ratio: "
	for i in range(1, 11):
		print str(i) + ": " + odds_list[i-1][0]
	print ""

if __name__ == "__main__":main()