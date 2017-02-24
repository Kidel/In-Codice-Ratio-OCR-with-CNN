from pipeline1 import pipeline1
from ocr_ensamble_builder import ocr_cnn_ensamble_builder
import dataset_generator as dataset
import numpy as np
import os

class pipeline3:

	def __init__(self, classes=dataset.ALPHABET_ALL, nb_epochs_ocr = 800, nb_epochs_cut_classifier=800, \
	 number_of_nets_ocr=5, number_of_nets_cut_classifier=5, batch_size=128, path_ocr="checkpoints/pipeline3", \
	 path_cut_classifier="checkpoints/pipeline1"):

		if not os.path.exists(path_ocr):
		    os.makedirs(path_ocr)

		if not os.path.exists(path_cut_classifier):
		    os.makedirs(path_cut_classifier)

		self._classes = classes

		self._cut_classifier = pipeline1(chars=classes, nb_epochs=nb_epochs_cut_classifier, \
			number_of_nets=number_of_nets_cut_classifier, batch_size=batch_size, path=path_pipeline1)

		self._ocr_net = ocr_cnn_ensamble_builder(len(classes), nb_epochs_ocr, number_of_nets=number_of_nets, path=path_ocr)

	def fit_ocr_net(self, X_train, y_train, X_test=[], y_test=[], verbose=0):
		self._ocr_net.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=verbose)

	def fit_letter_cut_classifier(self, letter, X_train, y_train, X_test=[], y_test=[], verbose=0):
		self._cut_classifier.fit_letter(letter, X_train, y_train, X_test=X_test, y_test=y_test, verbose=verbose)

	def predict(self, X_test, verbose=0):

		prediction_cuts = self._cut_classifier.predict(X_test)

		index_good_letters = []

		for i,(cut,_) in enumerate(prediction_cuts):
			if cut:
				index_good_letters.append(i)

		X_test = np.array(X_test)
		X_test_ocr = X_test[index_good_letters]
		prediction_ocr = self._ocr_net.predict(X_test_ocr)

		prediction = []

		ocr_i = 0;

		for i in enumerate(X_test):
			if not i in index_good_letters:
				prediction.append((False, []))
			else:
				 sorted_indexes = (-prediction_ocr[ocr_i]).argsort()[:3]
				 ranking = [(self._classes[j], prediction_ocr[ocr_i][j]*100) for j in sorted_indexes]
				 dt = np.dtype([('letters', np.str_, 16), ('grades', np.float64)])
				 ranking = np.array(ranking, dtype=dt)
				 prediction.append((True, ranking))
 				 ocr_i += 1

		return prediction
