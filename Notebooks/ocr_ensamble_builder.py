from ocr_cnn import OCR_NeuralNetwork
import ensemble

def ocr_cnn_ensamble_builder(classes, nb_epochs, number_of_nets=5, path="checkpoints//temp"):
	
	my_ensemble = ensemble.ensemble()

	for i in range(number_of_nets):
		net = OCR_NeuralNetwork(classes, nb_epochs=nb_epochs, model_dir=path, model_name=str(i), batch_size=128)
		my_ensemble.add_model(net)

	return my_ensemble





