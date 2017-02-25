from ocr_ensamble_builder import ocr_cnn_ensamble_builder
import dataset_generator as dataset
import numpy as np
import os


class pipeline2:

	def __init__(self, classes=dataset.ALPHABET_ALL, nb_epochs_ocr = 800, nb_epochs_cut_classifier=800, number_of_nets_ocr=5,\
	 number_of_nets_cut_classifier=5, batch_size=128, path_ocr="checkpoints/pipeline2", \
	 path_cut_classifier="checkpoints/cut_classifier", nb_filters1_cut=20, nb_filters2_cut=40, \
	 nb_filters1_ocr=20, nb_filters2_ocr=40, dense_layer_size1_cut=250, dense_layer_size1_ocr=250 ):

		if not os.path.exists(path_ocr):
		    os.makedirs(path_ocr)

		if not os.path.exists(path_cut_classifier):
		    os.makedirs(path_cut_classifier)

		self._classes = classes

		self._cut_classifier = ocr_cnn_ensamble_builder(2, nb_epochs_cut_classifier, number_of_nets=number_of_nets_cut_classifier,\
										 path=path_cut_classifier, nb_filters1=nb_filters1_cut, nb_filters2=nb_filters2_cut,\
										 dense_layer_size1=dense_layer_size1_cut)

		self._ocr_net = ocr_cnn_ensamble_builder(len(classes), nb_epochs_ocr, number_of_nets=number_of_nets_ocr, path=path_ocr,\
			nb_filters1=nb_filters1_ocr, nb_filters2=nb_filters2_ocr, dense_layer_size1=dense_layer_size1_ocr)


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

		for i,_ in enumerate(X_test):
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






