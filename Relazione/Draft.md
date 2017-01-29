# Realizzazione CNN per In Codice Ratio

Realizzazione di diverse CNN con tensorflow, libreria Keras, in ambiente Windows con supporto per GPU per Cuda cores (Nvidia).

* Installazione ambiente di sviluppo
* Realizzazione di un primo modello (standard keras) su dataset MNIST
* Studio del [MCDNN for image recognition](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf) di Dan Ciresan, Ueli Meier e Jurgen Schmidhuber: architettura DNN, combinazione di più colonne di DNN in una multi-colonna (approccio ensemble learning), processo di training con distorsione dell'immagine ad ogni epoca, funzioni di attivazione (tanh, lineare, softmax), dataset sperimentali (MNIST, NIST SD 19, CIFAR 10, caratteri cinesi, immagini stereo di oggetti 3d).
* Realizzazione di un modello basato sul [paper](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf) e confronto con activation function più recente.

## TO DO:
* introduzione di callback per metriche
* realizzazione di altri modelli per valutare diversi risultati
* adattamento dell'input da MNIST al dataset di In Codice Ratio
* esperimento con Transfer Learning da Inception Model
* realizzazione di una o più librerie
* test in ICR
