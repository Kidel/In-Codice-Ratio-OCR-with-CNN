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
