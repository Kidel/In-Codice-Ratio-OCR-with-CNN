import dataset_generator as dataset
import numpy as np
import os

class pipeline3:

	def __init__(self, cut_classifier, ocr_net, classes=dataset.ALPHABET_ALL):
		self._classes = classes
		self._cut_classifier = cut_classifier
		self._ocr_net = ocr_net



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

