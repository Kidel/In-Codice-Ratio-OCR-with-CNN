from ocr_ensamble_builder import ocr_cnn_ensamble_builder
import dataset_generator as dataset
import numpy as np
import os


class pipeline2:

	def __init__(self, classes=dataset.ALPHABET_ALL, nb_epochs_ocr = 800, nb_epochs_cut_classifier=800 number_of_nets=5,\
	 batch_size=128, path_ocr="checkpoints/pipeline2", path_cut_classifier="checkpoints/cut_classifier"):

		if not os.path.exists(path_ocr):
		    os.makedirs(path_ocr)

		if not os.path.exists(path_cut_classifier):
		    os.makedirs(path_cut_classifier)

		self._classes = classes

		self._cut_classifier = ocr_cnn_ensamble_builder(2, nb_epochs_cut_classifier, number_of_nets=number_of_nets, path=path_cut_classifier)
		self._ocr_net = ocr_cnn_ensamble_builder(len(classes), nb_epochs_ocr, number_of_nets=number_of_nets, path=path_ocr)


	def fit_ocr_net(self, X_train, y_train, X_test=[], y_test=[], verbose=0):
		self._ocr_net.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=verbose)

	def fit_cut_classifier(self, X_train, y_train, X_test=[], y_test=[], verbose=0):
		self._cut_classifier.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=verbose)

	# Return a list of tuples where the index i of the list represent the prediction
	# for the i-th value. Each tuple contains a boolean, True if is a good cut False otherwise,
	# and contains a list that is the ranking of the prediction for each letter. 
	# es: [(False, []), (True, [("s_alta",90%),("a", 10%),...("z", 0%)])]
	def predict(self, X_test, verbose=0):
		prediction_cuts = self._cut_classifier.predict(X_test, verbose=verbose)

		index_good_letters = []

		for i,(_,prob_letter) in enumerate(prediction_cuts):
			if prob_letter>=0.5:
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






