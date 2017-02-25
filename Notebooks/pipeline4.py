from pipeline1 import pipeline1
from ocr_ensamble_builder import ocr_cnn_ensamble_builder
import dataset_generator as dataset
import numpy as np
import os

class pipeline4:

	def __init__(self, cut_classifier, binary_nets, classes=dataset.ALPHABET_ALL ):

		if not os.path.exists(path_cut_classifier):
		    os.makedirs(path_cut_classifier)

		if not os.path.exists(path_pipeline1):
		    os.makedirs(path_pipeline1)

		self._classes = classes

		self._cut_classifier = cut_classifier
		#ocr_cnn_ensamble_builder(2, nb_epochs_cut_classifier, \
		#		number_of_nets=number_of_nets_cut_classifier, path=path_cut_classifier,\
		#		nb_filters1=nb_filters1_cut, nb_filters2=nb_filters2_cut, dense_layer_size1=dense_layer_size1_cut)

		self._pipeline1 = binary_nets
		#pipeline1(classes=classes, nb_epochs=nb_epochs_pip1, \
		#	number_of_nets=number_of_nets_pip1, batch_size=batch_size, path=path_pipeline1,\
		#	nb_filters1=nb_filters1_pip, nb_filters2=nb_filters2_pip, dense_layer_size1=dense_layer_size1_pip)


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



