import dataset_generator as dataset
import numpy as np
import os

class pipeline4:

	def __init__(self, cut_classifier, binary_nets, classes=dataset.ALPHABET_ALL ):

		self._classes = classes
		self._cut_classifier = cut_classifier
		self._binary_nets = binary_nets



	def predict(self, X_test, verbose=0):
		prediction_cuts = self._cut_classifier.predict(X_test, verbose=verbose)

		index_good_letters = []

		for i,(_,prob_letter) in enumerate(prediction_cuts):
			if prob_letter>=0.5:
				index_good_letters.append(i)

		X_test = np.array(X_test)
		X_test_pip1 = X_test[index_good_letters]

		prediction_pip1 = self._binary_nets.predict(X_test_pip1)

		predictions = []

		pip_index = 0

		for i,_ in enumerate(X_test):
			if not i in index_good_letters:
				predictions.append((False, []))
			else:
				predictions.append(prediction_pip1[pip_index])
				pip_index += 1

		return predictions



