from pipeline1 import pipeline1
from ocr_ensamble_builder import ocr_cnn_ensamble_builder
import dataset_generator as dataset
import numpy as np
import os

class pipeline4:

	def __init__(self, classes=dataset.ALPHABET_ALL, nb_epochs_pip1 = 800, nb_epochs_cut_classifier=800, \
	 number_of_nets_pip1=5, number_of_nets_cut_classifier=5, batch_size=128, path_cut_classifier="checkpoints/cut_classifier", \
	 path_pipeline1="checkpoints/pipeline1", nb_filters1_cut=20, nb_filters2_cut=40, \
	 nb_filters1_pip=20, nb_filters2_pip=40, dense_layer_size1_pip=150, dense_layer_size1_cut=250 ):

		if not os.path.exists(path_cut_classifier):
		    os.makedirs(path_cut_classifier)

		if not os.path.exists(path_pipeline1):
		    os.makedirs(path_pipeline1)

		self._classes = classes

		self._cut_classifier = ocr_cnn_ensamble_builder(2, nb_epochs_cut_classifier, \
				number_of_nets=number_of_nets_cut_classifier, path=path_cut_classifier,\
				nb_filters1=nb_filters1_cut, nb_filters2=nb_filters2_cut, dense_layer_size1=dense_layer_size1_cut)

		self._pipeline1 = pipeline1(classes=classes, nb_epochs=nb_epochs_pip1, \
			number_of_nets=number_of_nets_pip1, batch_size=batch_size, path=path_pipeline1,\
			nb_filters1=nb_filters1_pip, nb_filters2=nb_filters2_pip, dense_layer_size1=dense_layer_size1_pip)


	def fit_letter_pipeline1(self, letter, X_train, y_train, X_test=[], y_test=[], verbose=0):
		self._pipeline1.fit_letter(letter, X_train, y_train, X_test=X_test, y_test=y_test, verbose=verbose)

	def fit_cut_classifier(self, X_train, y_train, X_test=[], y_test=[], verbose=0):
		self._cut_classifier.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=verbose)

	def predict(self, X_test, verbose=0):
		prediction_cuts = self._cut_classifier.predict(X_test, verbose=verbose)

		index_good_letters = []

		for i,(_,prob_letter) in enumerate(prediction_cuts):
			if prob_letter>=0.5:
				index_good_letters.append(i)

		X_test = np.array(X_test)
		X_test_pip1 = X_test[index_good_letters]

		prediction_pip1 = self._pipeline1.predict(X_test_pip1)

		predictions = []

		pip_index = 0

		for i,_ in enumerate(X_test):
			if not i in index_good_letters:
				predictions.append((False, []))
			else:
				predictions.append(prediction_pip1[pip_index])
				pip_index += 1

		return predictions


	# Take in input an array of images, an array of labels, and returns the precision 
	# of the classifier
	def evaluate(self, X_test, y_test):
		prediction = self.predict(X_test)

		score = 0
		not_a_letter_count = 0

		for i,(is_a_letter,ranking) in enumerate(prediction):
			if is_a_letter:
				if ranking[0][0] == self._classes[y_test[i]]:
					score += 1
			else:
				not_a_letter_count += 1

		return score/(len(X_test)-not_a_letter_count)


