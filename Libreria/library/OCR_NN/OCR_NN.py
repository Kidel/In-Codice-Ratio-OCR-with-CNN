import numpy as np
import os.path
from util import Util
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from keras.datasets import mnist


class OCR_NeuralNetwork:
    
    # Plot utils
    _u = Util()
    
    # Model
    _model = None
    _activation = "relu"

    _batch_size = 0;
    _nb_classes = 0;
    _nb_epochs = 0;
    _model_path = "";
        
    # input image dimensions
    _img_rows, _img_cols = 28, 28 
    _input_shape = (_img_rows, _img_cols, 1)
    
    # number of convolutional filters to use
    _nb_filters1 = 20
    _nb_filters2 = 40

    # size of pooling area for max pooling
    _pool_size1 = (2, 2)
    _pool_size2 = (3, 3)

    # convolution kernel size
    _kernel_size1 = (4, 4)
    _kernel_size2 = (5, 5)

    # dense layer size
    _dense_layer_size1 = 150

    # dropout rate
    _dropout = 0.15

    # Model first loaded
    _loaded = False
    
    def __init__(self, nb_classes, nb_epochs=50, model_path = "checkpoints/temp.hdf5", batch_size=128, activation="relu"):
        self._batch_size = batch_size
        self._nb_classes = nb_classes
        self._nb_epochs = nb_epochs
        self._model_path = os.path.join(model_path, 'temp.hdf5') 
        self._activation = activation
         
    def _init_model(self):
        self._model = Sequential()
        self._model.add(Convolution2D(self._nb_filters1, self._kernel_size1[0], self._kernel_size1[1],
                                border_mode='valid',
                                input_shape=self._input_shape, name='covolution_1_' + str(self._nb_filters1) + '_filters'))
        self._model.add(Activation(self._activation, name='activation_1_' + self._activation))
        self._model.add(MaxPooling2D(pool_size=self._pool_size1, name='max_pooling_1_' + str(self._pool_size1) + '_pool_size'))
        self._model.add(Convolution2D(self._nb_filters2, self._kernel_size2[0], self._kernel_size2[1]))
        self._model.add(Activation(self._activation, name='activation_2_' + self._activation))
        self._model.add(MaxPooling2D(pool_size=self._pool_size2, name='max_pooling_1_' + str(self._pool_size2) + '_pool_size'))
        self._model.add(Dropout(self._dropout))

        self._model.add(Flatten())
        self._model.add(Dense(self._dense_layer_size1, name='fully_connected_1_' + str(self._dense_layer_size1) + '_neurons'))
        self._model.add(Activation(self._activation, name='activation_3_' + self._activation))
        self._model.add(Dropout(self._dropout))
        self._model.add(Dense(self._nb_classes, name='output_' + str(self._nb_classes) + '_neurons'))
        self._model.add(Activation('softmax', name='softmax'))

        self._model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy', 'precision', 'recall', 'mean_absolute_error'])

    def fit(self, X_train, y_train, X_test, y_test, forceRetrain = True, verbose = 0):

        if(forceRetrain):
            ## delete only if file exists ##
            if os.path.exists(self._model_path):
                os.remove(self._model_path)
            else:
                print("Older Nerual Net could not be found, creating a new net...")

            # Initialize the model
            self._init_model()

        # If it is not loaded yet, try load it from fs and create a new model
        elif not self._loaded:
            self._init_model()
            self._try_load_model_from_fs()

            _loaded = True


        X_train,y_train = self._preprocess_data(X_train, y_train)
        X_test, y_test = self._preprocess_data(X_test, y_test)
        
        # checkpoint
        checkpoint = ModelCheckpoint(self._model_path, monitor='val_acc', verbose=verbose, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # training
        print('training relu model')
        history = self._model.fit(X_train, y_train, batch_size=self._batch_size, nb_epoch=self._nb_epochs,
                  verbose=verbose, validation_data=(X_test, y_test), callbacks=callbacks_list)
        
        return history

        
    def _preprocess_data(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], self._img_rows, self._img_cols, 1)
        
        X = X.astype('float32')
        X /= 255
        
        print('X shape:', X.shape)
        print(X.shape[0], 'train samples')
        
        # convert class vectors to binary class matrices
        y = np_utils.to_categorical(y, self._nb_classes)
        
        return (X,y)
    
    def _try_load_model_from_fs(self):
        
        # loading weights from checkpoints 
        if os.path.exists(self._model_path):
            self._model.load_weights(self._model_path)
        else:
            print("Previous Model Not Found")
        
    def evaluate(self, X_test, y_test, verbose = 0):
        X, y = self._preprocess_data(X_test, y_test)
        score = self._model.evaluate(X, y, verbose = verbose)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score
        
    def predict(self, X_test):
        X_test = X_test.reshape(X_test.shape[0], self._img_rows, self._img_cols, 1)
        X_test = X_test.astype('float32')
        X_test /= 255  
        #X_test, _ = preprocess_data(X_test, [])
        return self._model.predict_classes(X_test)



def main():
    ## Fast Usage
    
    # Prepare the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Initialization
    nn = OCR_NeuralNetwork(10, nb_epochs=1, model_path="checkpoints", batch_size=128)

    # Training
    history = nn.fit(X_train, y_train, X_test, y_test, forceRetrain = False)

    # Prediciton
    predicted = nn.predict(X_test)

    # Evaluation
    score = nn.evaluate(X_test, y_test, verbose=1)


# Execute the module if it is main
if __name__ == "__main__":
    main()
