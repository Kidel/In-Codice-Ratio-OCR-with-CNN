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
- Quarto modello:
 - Applicazione della tecnica ensemble learning alla rete multicolonna. Miglior prestazione ottenuta: 0.4 error rate.
- Realizzazione di una o più librerie:
 - libreria ```OCR_NN``` che espone funzioni di inizializzazione, training, valutazione e classificazione.
 - libreria util per il plotting di grafici e metriche.
 
### test su ICR
- Classificatori binari per singolo carattere 
 - test su dataset ICR
- Modello a 22 classi
 - confronto con i modelli a lettera singola (necessario merge)
 - basso mae e alta precision
 - ~~utilizzare le reti binarie per segmentare o una rete segmentatrice?~~
- Rete segmentatrice
- Test su pipeline
 - Merge dei classificatori binari
 - Segmentatrice + Modello 22 classi
 - Classificatori binari + Modello 22 classi

## TO DO:
* introduzione di callback per metriche
* ~~realizzazione di altri modelli per valutare diversi risultati~~
* ~~**valutare risultati del terzo modello di cnn**~~
* ~~adattamento dell'input da MNIST al dataset di In Codice Ratio~~
* ~~esperimento con Transfer Learning da Inception Model~~
* test in ICR

# Il problema

**In Codice Ratio** è un progetto curato dall'*Università degli Studi di Roma Tre* in collaborazione con l'*Archivio Segreto dello Stato del Vaticano*. Tale progetto ha lo scopo di digitalizzare i documenti e i testi antichi contenuti nell'Archivio.

Il problema che abbiamo affrontato è dunque quello di classificare le lettere scritte a mano (in scrittura carolina) a partire dalla loro immagine opportunamente estratta, al fine di riconoscerle. L'input sarà un insieme di possibili tagli delle parole da leggere. Il nostro sistema dovrà essere in grado non solo di riconoscere le lettere contenute in un buon taglio, ma anche di scartare i tagli errati non riconducibili ad alcuna lettera.

Per affrontare tale problema abbiamo scelto di utilizzare il Deep Learning, costruendo più Convolutional Neural Network, studiando le migliori architetture allo stato dell'arte e riadattandole per questo contesto.

# L'ambiente

Il sistema si basa sulla libreria open-source [**TensorFlow**](https://www.tensorflow.org/), gestita ad alto livello tramite la libreria [*Keras*](https://keras.io/).
Per ottenere migliori prestazioni, utilizziamo la versione di Tensorflow per GPU, con supporto per Cuda core (cuDNN v5.1).

Gli esperimenti sono stati svolti su 3 macchine diverse messe a disposizione dai membri del gruppo, tutte e tre in ambiente Windows con processori Intel e GPU Nvidia dotate rispettivamente di 384, 1024 e 1664 Cuda core. La prima GPU con 16 GB di shared memory, le altre 2 con rispettivamente 2 GB e 4 GB di dedicated memory. 

Windows è stato preferito ad un sistema Unix per la miglior gestione del carico di lavoro sulla GPU, dal momento che è possibile escludere la UI del sistema operativo dalla GPU dedicata e farla elaborare alla GPU integrata sui processori Intel. Il codice utilizza comunque librerie compatibili con Linux e Mac OS. 

## Il primo modello di CNN e il test dell'ambiente di lavoro

Il primo modello di CNN prodotto è una rete piuttosto semplice, impostata per verificare il corretto funzionamento del sistema.
Si tratta di una rete a 2 livelli convoluzionali e 2 hidden layer. 

# Multi-column Deep Neural Network

I modelli successivi si basano principalmente sullo studio del 2012 condotto da Dan Ciresan, Ueli Meier e Jurgen Schmidhuber ed esposto nel paper [Multi-column Deep Neural Network for image recognition](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf). 

A seguito di un attento studio, abbiamo sperimentato il loro approccio e riadattato alla luce delle nuove evidenze emerse nel campo delle reti neurali: in primo ruolo [DropConnect](http://cs.nyu.edu/~wanli/dropc/) e la funzione di attivazione Rectifier.

## La rete MCDNN

Il modello è costituito da diverse *colonne* di Deep Neural Network, tutte con la stessa struttura, le cui predizioni vengono successivamente combinate in una semplice media aritmetica (approccio **ensemble learning**).

<img src="images/MCDNN.png" alt="MCDNN" width="450"/>

La singola DNN ha la seguente struttura: 

![Simple DNN](images/simpleDNN.png)

Il training parte da pesi inizializzati randomicamente, e la tecnica del max pooling permette di determinare facilmente i neuroni più attivi per ogni regione di input. I neuroni così "selezionati" proseguono nell'allenamento, mentre gli altri non subiscono ulteriori correzioni nei pesi.

Le immagini di input vengono preprocessate a monte del training, e successivamente vengono distorte attraverso operazioni di rotazione, traslazione e scaling all'inizio di ogni epoca di addestramento, sempre in maniera randomica. Le immagini originali vengono invece usate in fase di validazione. Il preprocessamento può inoltre essere diverso per ogni colonna, in modo da ridurre sia il tasso d'errore che il numero di colonne necessarie a raggiungere un certo livello di accuracy.

Il training termina quando l'errore sul validation set arriva a zero, o quando il tasso di apprendimento raggiunge un minimo prestabilito.

Il modello ha una forte base biologica, ispirato alle reali connessioni presenti tra la retina e la corteccia visiva nei mammiferi.

### Esperimenti

Il modello è stato testato su diversi dataset conosciuti, con opportuni adattamenti determinati dalla natura del dataset stesso (dimensione, formato, ecc...).

L'esperimento sul quale ci confrontiamo è proprio quello su dataset **MNIST**, i cui risultati sono il nostro punto di riferimento.

Il dataset originale viene normalizzato rispetto a 7 diverse dimensioni in pixel, generando così 7 diversi dataset. Per ognuno di essi, si allena una rete multicolonna formata da 5 DNN semplici, ottenendo una rete multicolonna finale di 35 DNN. 
L'allenamento si svolge per 800 epoche. Ogni DNN impiega all'incirca 14 ore di addestramento, con scarsi miglioramenti oltre le 500 epoche.

Il *tasso d'errore* (definito banalmente come 1-precision) è dello 0.23% per la rete multicolonna, il che rappresenta un enorme traguardo per lo stato dell'arte nel 2012, battuto solo di recente.

Per quanto riguarda la singola DNN, il *tasso d'errore* sale allo **0.4%**: sarà questo il nostro punto di riferimento.

## Il secondo modello di CNN

Siamo quindi partiti dall'architettura proposta nel paper, e abbiamo apportato alcune modifiche dettate dalle nuove evidenze emerse negli ultimi anni.

La prima sostanziale modifica al modello è stata quella di aggiungere due livelli di **Dropout**. Questi livelli ci permettono di settare una certa frazione casuale delle unità di input a 0 ad ogni iterazione della fase di training, evitando così di incorrere in overfitting.

La seconda riguarda la funzione di attivazione. Nel modello multicolonna viene utilizzata la *tangente iperbolica*, simile alla sigmoide logistica ma in range [-1, 1] invece di [0, 1]. Il nostro esperimento è stato quello di sostituire questa funzione con la **rectifier**. I vantaggi di questa sostituzione sono diversi: la ReLU è molto semplice da calcolare (richiede solo addizioni, moltiplicazioni e confronti), permette un'attivazione sparsa dei neuroni (evita l'overfitting), e una migliore propagazione dei gradienti. Semplicemente, risulta più veloce ed efficiente della sigmoide per architetture deep e dataset complessi.

Il secondo modello di CNN è dunque il seguente:

![Secondo modello CNN](images/neural-network-icr.png)

### Valutazione su dataset MNIST

I risultati ottenuti hanno confermato quanto detto: la funzione ReLU risulta più efficiente e ci concede migliori valori di accuracy (0.9934 contro 0.9906 ottenuto da tanh). 

Il numero di epoche è invece risultato eccessivo, portando la rete a fare overfitting. Alla luce di ulteriori test, un numero migliore sarebbe intorno alle 50. Anche in questo caso, la ReLU ha mostrato un comportamento migliore rispetto tanh, che invece degrada già intorno alle 40 epoche. Visto il guadagno in precision ottenuto con 800 epoche, rispetto al tempo necessario, riteniamo che sia poco conveniente.

Il *tasso d'errore*, in fine, è dello **0.55%**: risulta migliore rispetto al primo modello di rete, ma ancora non raggiunge il valore di riferimento.

## Il terzo modello con distorsione

Seguendo l'esempio della MCDNN, abbiamo deciso di introdurre una **fase di distorsione** delle immagini all'inizio di ogni epoca di training. Le operazioni di distorsione applicate sono della stessa tipologia di quelle usate per la MCDNN, ovvero si tratta di rotazioni (in una finestra di 40°), traslazioni (verticali e orizzontali) e scaling (del 10%), eseguite randomicamente a partire da un seed.

Il training viene eseguito inoltre in due tempi: inizialmente sulle immagini distorte, e in un secondo tempo sulle immagini originali, per avere qualche bias sulle immagini non deformate.

Come ulteriore esperimento, abbiamo deciso di modificare la dimensione dell'hidden layer, portandolo da 150 a 200, supponendo che tale dimensione fosse dettata da motivi computazionali.

### Valutazione

La divisione dell'addestramento in due fasi, tra immagini distorte e immagini originali, ci mostra già buoni risultati: notiamo infatti che tra la prima e la seconda fase si ha un incremento dello 0.15% in precision.

L'addestramento richiede chiaramente tempi più lunghi con questa architettura: notiamo infatti una dilatazione dei tempi per ogni epoca, passando da 6 secondi a 20 sulla nostra GPU media (GTX970). Il tempo totale per completare il training è stato di *3 ore*.
Ciò nonostante, il *tasso d'errore* scende considerevolemente da 0.55 a **0.45**, a parità di numero di epoche.

L'aumento della dimensione dell'hidden layer inoltre ha portato un lieve miglioramento del tasso d'errore, raggiungendo lo **0.44**.

Si tratta di un buono risultato, molto vicino a quanto ottenuto dalla DNN semplice del paper.

Resta comunque aperta la questione della convenienza, ovvero se valga la pena di attendere un addestramento di 3 ore, rispetto ai 4 minuti della rete precedente, per ottenere un incremento di precision dello 0.10%.

## Il quarto modello multi colonna

In questo modello si applica la tecnica dell'**ensemble learning**, così da avvicinarsi all'architettura completa esposta nel paper.

Per semplicità, abbiamo costruito una rete di 5 colonne anche se, come già spiegato, nel paper viene costruita una rete più complessa, con 7 reti da 5 colonne, addestrate su diverse alterazioni del dataset. Il fine dell'esperimento è comunque quello di capire se  possiamo trarre un effettivo vantaggio dall'ensemble learning, per cui va bene utilizzare una rete più piccola ma che ci permette di comprimere i tempi di addestramento.

### Valutazione

I tempi di addestramento sono proporzionali al numero di reti che costituiscono la multi colonna, non avendo la possibilità di parallelizzare il processo sul nostro attuale hardware. Il tempo impiegato è stato dunque di 12 ore.

Il *tasso d'errore* è migliorato, raggiungendo lo **0.4**. Si tratta del miglior risultato ottenuto, e ci conferma quanto espresso dal paper, ovvero che la tecnica dell'ensemble learning offre effetivamente prestazioni migliori. Notiamo inoltre che abbiamo ottenuto un risultato analogo rispetto alla 5 MCDNN del paper che lavorando sulle immagini in dimensione originale, la quale ottiene un tasso d'errore identico.

Aggiungendo un numero adeguato di colonne e con le opportune trasformazioni del dataset sembra quindi essere possibile raggiungere i risultati pubblicati dal paper, ovvero quello 0.23% che tanto si avvicina all'errore umano dello 0.2%.

# Esperimenti sul dataset di In Codice Ratio
Per affrontare il problema proposto abbiamo costruito e sperimentato su 3 diverse architetture. Di seguito riportiamo i modelli e i risultati degli esperimenti.

## Classificatori binari per singolo carattere
L'architettura a 5 colonne con layer 30C-50C-200N è stata usata per condurre due diversi esperimenti sulla costruzione del training set, con un rapporto tra esempi positivi e negativi prima di 1:1 e poi di 1:2.

Dalla **valutazione** sono risultati dei livelli d'accuracy e tassi d'errore distribuiti piuttosto uniformemente tra i diversi caratteri, quasi sempre sotto l'8% d'errore, sebbene troviamo alcune significative eccezioni.
Lettere particolarmente difficili da distinguere sono state la **i**, la **m**, la **n**, la **u** e la **h**. Intuiamo che buona parte del problema, in generale anche per le altre lettere, sia posto nell'etichettatura del dataset, che contiene diversi errori commessi nella fase di crowdsourcing: vediamo infatti negli esempi di classificazione incorrette che molto spesso si trattava di immagini riconosciute correttamente dalla rete ma etichettate male dal dataset. Tuttavia la situazione si aggrava per lettere facilmente confondibili tra loro, che sono proprio la i, la m, la n e la u, che nella scrittura carolingia appaiono quasi come una concatenazione di i, o di corte linee verticali, distinguibili per lo più dal contesto e nel migliore dei casi dalla legatura del carattere alle lettere successive e precedenti. Per la **h** il problema è stato posto soprattutto dalla scarsità di esempi (circa 60).
Di seguito riportiamo i tassi d'errore relativi alle lettere problematiche:

<div style="align:center">

Carattere | Ratio pos:neg 1:1 | Ratio pos:neg 1:2
----------|-------------------|------------------
     i    |       17,5%       |       17,5% 
     m    |        9,9%       |       10.3%
     n    |       12,3%       |       12.9%
     u    |         16%       |       15.8%  
s_mediana |        3.3%       |        6.6%
     h    |      22,22%       |       13.8%
     f    |          5%       |        7.5%
     
</div>
   
La tabella ci mostra come il tasso d'errore cambi in positivo o in negativo in base alla lettera e alla ratio del training set. Abbiamo inoltre calcolato l'errore medio commesso da tutti i classificatori allenati sui due diversi training set: per il rapporto **1:1** abbiamo un **tasso d'errore medio** del **7,5%**, mentre per il rapporto **1:2** del **7,1%**, influenzato probabilmente dal netto miglioramento dell'errore sulla h. 
Le cause di questa altalenanza sono da ricercarsi probabilmente nell'etichettatura del dataset, che in certi casi mostra tagli errati come buoni esempi. L'ambiguità del dataset porta all'incostanza della classificazione, per cui per un migliore allenamento è necessario un dataset ripulito.

## Classificatore Multiclasse a 22 classi (OCR)
Questo modello si basa sull'architettura a 5 colonne con layer 50C-100C-250N e si tratta di un'unica rete a 22 classi. Poiché svolge un diverso task da quello dei classificatori binari, che distinguono le singole lettere dalle altre e dal rumore, questa rete non è utilizzabile da sola per risolvere il problema posto, ma verrà inserita in una pipeline per raggiungere lo scopo desiderato.

La rete raggiunge il **95,1%** di **accuracy** e il **94,9%** di **recall**, con un **tasso d'errore** del **4,6%**. Ispezionando la matrice di confusione, si nota il buon comportamento della rete con poche eccezioni. I caratteri **h** e **f** vengono spesso scambiate rispettivamente con i caratteri **b** e **s alta**. Di seguito riportiamo i valori:

<div style="align:center">

Carattere |         h         |         b
----------|-------------------|------------------
     h    |         3         |         13 
     b    |         0         |         80

Carattere |         f         |       s_alta
----------|-------------------|------------------
     f    |        14         |          5 
  s_alta  |        23         |         90
  
</div>

L'errore sembra nascere dalla carenza di buoni esempi di h e f nel dataset, per cui la rete tende a confondere queste lettere con i due caratteri che più somigliano. Non è da escludere la presenza di errori nell'etichettatura effettuata dal crowdsourcing, soprattutto per i caratteri s e f che sono molto simili.

In generale ci aspettiamo che, come per i classificatori binari, una pulitura del dataset possa portare ad ancora migliori prestazioni e a risolvere questi casi particolari.

## Classificatore binario dei tagli
Questo classificatore è stato pensato per essere usato in serie con la rete multiclasse e ha il compito di stabilire se un dato segmento rappresenta un buon taglio o no. Il training set è stato costruito a partire da un'unificazione del dataset delle lettere tagliate bene (formando la classe "good") con un dataset aggiuntivo di tagli troppo grandi o troppo piccoli per ogni carattere (formando la classe "wrong").
Abbiamo eseguito due esperimenti con due diverse architetture, una analoga a quella dei classificatori binari per singolo carattere (30C-50C-200N) ed una analoga a quella del classificatore multiclasse (50C-100C-250N).

Il primo esperimento ha raggiunto un'accuracy del **93,4%** ed un tasso d'errore del **6,5%**. Questo risultato è stato raggiunto già dalla singola colonna del secondo esperimento, e ciò ci lascia intuire che il numero di caratteristiche da estrarre per questo task è più elevato.


**TODO**

## Pipeline
