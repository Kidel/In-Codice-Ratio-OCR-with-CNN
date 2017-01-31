# Realizzazione CNN per In Codice Ratio

Realizzazione di diverse CNN con tensorflow, libreria Keras, in ambiente Windows con supporto per GPU per Cuda cores (Nvidia).

- Installazione ambiente di sviluppo
- Realizzazione di un primo modello (standard keras) su dataset MNIST
- Studio del [MCDNN for image recognition](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf) di Dan Ciresan, Ueli Meier e Jurgen Schmidhuber: architettura DNN, combinazione di più colonne di DNN in una multi-colonna (approccio ensemble learning), processo di training con distorsione dell'immagine ad ogni epoca, funzioni di attivazione (tanh, lineare, softmax), dataset sperimentali (MNIST, NIST SD 19, CIFAR 10, caratteri cinesi, immagini stereo di oggetti 3d).
- Studio di [Regularization of Neural Networks using DropConnect](http://cs.nyu.edu/~wanli/dropc/).
- Realizzazione di un modello basato sul [paper](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf) e [DropConnect](http://cs.nyu.edu/~wanli/dropc/) e confronto con activation function più recente:
 - Funzione di attivazione **relu** è risultata migliore di tanh (0.9934 vs 0.9906 accuracy nelle stesse condizioni e durante tutte le epoche).
 - Confronto tra [algoritmi di ottimizzazione](http://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html) su MNIST.
 - 800 epoche risultano in overfitting, un numero migliore è nell'ordine di 50-650.
 - Errore intorno allo 0.55%.
- Terzo modello di CNN:
 - Introduzione di distorsione randomica delle immagini: rotazione in una finestra di 30°, shift verticale e orizzontale e zoom in un range del 10%. Sono state testate finestre maggiori e minori di 30° e presenza o assenza di shift e zoom. E' stata testata la presenza o meno di epoche con dataset pulito. 25 epoche alla fine del precedente training sono risultate positive per la precision senza portare a overfitting.
- Realizzazione di una o più librerie:
 - libreria ```OCR_NN``` che espone funzioni di inizializzazione, training, valutazione e classificazione.
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

Gli esperimenti sono stati svolti su 3 macchine diverse messe a disposizione dai membri del gruppo, tutte e tre in ambiente Windows con processori Intel e GPU Nvidia dotate rispettivamente di 384, 1024 e 1664 Cuda core. La prima GPU con 16 GB di shared memory, le altre 2 con rispettivamente 2 GB e 4 GB di dedicated memory. 

Windows è stato preferito ad un sistema Unix per la miglior gestione del carico di lavoro sulla GPU, dal momento che è possibile escludere la UI del sistema operativo dalla GPU dedicata e farla elaborare alla GPU integrata sui processori Intel. Il codice utilizza comunque librerie compatibili con Linux e Mac OS. 

## Il primo modello di CNN e test dell'ambiente di lavoro

Il primo modello di CNN prodotto è una rete piuttosto semplice, impostata per verificare il corretto funzionamento del sistema.
Si tratta di una rete a 2 livelli convoluzionali e 2 hidden layer. 

**Immagine rete**

# Multi-column Deep Neural Network e il secondo modello di CNN

Il secondo modello si basa sullo studio del 2012 condotto da Dan Ciresan, Ueli Meier e Jurgen Schmidhuber ed esposto nel paper [Multi-column Deep Neural Network for image recognition](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf). 

A seguito di un attento studio, abbiamo sperimentato il loro approccio e riadattato alla luce delle nuove evidenze emerse nel campo delle reti neurali (in primo ruolo [DropConnect](http://cs.nyu.edu/~wanli/dropc/) e la funzione di attivazione Rectifier).

## Multi-column Deep Neural Network
### La rete

Il modello è costituito da diverse *colonne* di Deep Neural Network, tutte con la stessa struttura, le cui predizioni vengono successivamente combinate in una semplice media aritmetica (approccio **ensemble learning**).

**Immagine multi-colonna**

La singola DNN ha la seguente struttura: 

**Immagine rete**

Il training parte da pesi inizializzati randomicamente, e la tecnica del max pooling permette di determinare facilmente i neuroni più attivi per ogni regione di input. I neuroni così "selezionati" proseguono nell'allenamento, mentre gli altri non subiscono ulteriori correzioni nei pesi.

Le immagini di input vengono preprocessate a monte del training, e successivamente vengono distorte attraverso operazioni di rotazione, traslazione e scaling all'inizio di ogni epoca di addestramento, sempre in maniera randomica. Le immagini originali vengono invece usate in fase di validazione. Il preprocessamento può inoltre essere diverso per ogni colonna, in modo da ridurre sia il tasso d'errore che il numero di colonne necessarie a raggiungere un certo livello di accuracy.

Il training termina quando l'errore sul validation set arriva a zero, o quando il tasso di apprendimento raggiunge un minimo prestabilito.

Il modello ha una forte base biologica, ispirato alle reali connessioni presenti tra la retina e la corteccia visiva nei mammiferi.

### Esperimenti

Il modello è stato testato su diversi dataset conosciuti, con opportuni adattamenti determinati dalla natura del dataset stesso (dimensione, formato, ecc...).

L'esperimento che è stato per noi di riferimento è proprio quello su dataset **MNIST**, i cui risultati sono il nostro punto di riferimento.

Il dataset originale viene normalizzato rispetto a 7 diverse dimensioni in pixel, generando così 7 diversi dataset. Per ognuno di essi, si allena una rete multicolonna formata da 5 DNN semplici, ottenendo una rete multicolonna finale di 35 DNN semplici. 
L'allenamento si svolge per 800 epoche. Ogni DNN impiega all'incirca 14 ore di addestramento, con scarsi miglioramenti oltre le 500 epoche.

Il *tasso d'errore* (definito banalmente come 1-precision) è dello 0.23% per la rete multicolonna, il che rappresenta un enorme traguardo per lo stato dell'arte nel 2012, battuto solo di recente.

Per quanto riguarda la singola DNN, il *tasso d'errore* sale allo **0.52%**: sarà questo il nostro punto di riferimento.

## Il secondo modello di CNN

Siamo quindi partiti dall'architettura proposta nel paper, e abbiamo apportato alcune modifiche dettate dalle nuove evidenze emerse negli ultimi anni.

La prima sostanziale modifica al modello è stata quella di aggiungere due livelli di **Dropout**. Questi livelli ci permettono di settare una certa frazione casuale delle unità di input a 0 ad ogni iterazione della fase di training, evitando così di incorrere in overfitting.

La seconda riguarda la funzione di attivazione. Nel modello multicolonna viene utilizzata la *tangente iperbolica*, simile alla sigmoide logistica ma in range [-1, 1] invece di [0, 1]. Il nostro esperimento è stato quello di sostituire questa funzione con la **rectifier**. I vantaggi di questa sostituzione diversi: la ReLU è molto semplice da calcolare (richiede solo addizioni, moltiplicazioni e confronti), permette un'attivazione sparsa dei neuroni (evita l'overfitting), e una migliore propagazione dei gradienti. Semplicemente, risulta più veloce ed efficiente della sigmoide per architetture deep e dataset complessi.

Il secondo modello di CNN è dunque il seguente:

**immagine**

## Valutazione su dataset MNIST

