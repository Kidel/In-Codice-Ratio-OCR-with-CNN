# Realizzazione CNN per In Codice Ratio

Realizzazione di diverse CNN con tensorflow, libreria Keras, in ambiente Windows con supporto per GPU per Cuda cores (Nvidia).

- Installazione ambiente di sviluppo
- Realizzazione di un primo modello (standard keras) su dataset MNIST
- Studio del [MCDNN for image recognition](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf) di Dan Ciresan, Ueli Meier e Jurgen Schmidhuber: architettura DNN, combinazione di più colonne di DNN in una multi-colonna (approccio ensemble learning), processo di training con distorsione dell'immagine ad ogni epoca, funzioni di attivazione (tanh, lineare, softmax), dataset sperimentali (MNIST, NIST SD 19, CIFAR 10, caratteri cinesi, immagini stereo di oggetti 3d).
- Realizzazione di un modello basato sul [paper](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf) e confronto con activation function più recente:
 - Funzione di attivazione **relu** è risultata migliore di tanh (0.9934 vs 0.9906 accuracy).
 - 800 epochs risultano in overfitting, un numero migliore è nell'ordine di 50.
 - Errore ancora troppo alto, intorno allo 0.50%.
- Terzo modello di CNN:
 - Introduzione di distorsione randomica delle immagini: rotazione in una finestra di 30°, shift verticale e orizzontale in un range del 10%.
- Realizzazione di una o più librerie:
 - libreria OCR_NN che espone funzioni di inizializzazione, training, valutazione e classificazione.
 - libreria util per il plotting di grafici e metriche.

## TO DO:
* introduzione di callback per metriche
* ~~realizzazione di altri modelli per valutare diversi risultati~~
* **valutare risultati del terzo modello di cnn**
* ~~adattamento dell'input da MNIST al dataset di In Codice Ratio~~
* esperimento con Transfer Learning da Inception Model
* test in ICR

# Problema

# Ambiente

Il sistema si basa sulla libreria open-source [**TensorFlow**](https://www.tensorflow.org/), gestita ad alto livello tramite la libreria [*Keras*](https://keras.io/).
Per ottenere migliori prestazioni, utilizziamo la versione di Tensorflow per GPU, con supporto per Cuda core (cuDNN v5.1).

Gli esperimenti sono stati svolti su 3 macchine diverse, in ambiente Windows e GPU Nvidia dotate rispettivamente di 384, 1024 e 1664 Cuda core.

## Il primo modello di CNN e test dell'ambiente di lavoro

Il primo modello di CNN prodotto è una rete piuttosto semplice, impostata per verificare il corretto funzionamento del sistema.
Si tratta di una rete a 2 livelli convoluzionali e 2 hidden layer. 

**Immagine rete**

