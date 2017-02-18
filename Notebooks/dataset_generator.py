import samples_repository
import numpy as np
import matplotlib.pyplot as plt

ALPHABET_LOW = ["g","p","q","s_bassa"]
ALPHABET_HIGH = ["d_alta","s_alta","b","f","h","l"]
ALPHABET_MID = ["a","c","d_mediana","e","i","m","n","o","r","s_mediana","t","u"]
ALPHABET_ALL = ["a","c","d_mediana","e","i","m","n","o","r","s_mediana","t","u",\
                "d_alta","s_alta","b","f","h","l","g","p","q","s_bassa"]

pos_neg_ratio=1.0
train_test_ratio=6.0/7.0
neg_labeled_height_ratio = 1.0/2.0

def split_by_ratio(array, ratio):
    first_half_size, second_half_size = calc_ratio(len(array), ratio)
    first_half, second_half = split(array, first_half_size)
    assert len(first_half) == first_half_size
    assert len(second_half) == second_half_size
    assert (len(first_half) + len(second_half)) == len(array)
    return (first_half, second_half)

def calc_ratio(val, ratio):
    first_half_size = int(val*ratio)
    second_half_size = val - first_half_size
    assert (first_half_size + second_half_size) == val
    return (first_half_size, second_half_size)

def split(array, index):
    first_half = array[:index]
    second_half = array[index:]
    return (first_half, second_half)


def generate_positive_and_negative_labeled(char, pos_neg_ratio=pos_neg_ratio, train_test_ratio=train_test_ratio, verbose=1):
    # Calcolo i samples e le labels positivi per il training e il test
    positive_samples = samples_repository.get_all_positive_samples_by_char(char)
    positive_samples_train, positive_samples_test = split_by_ratio(positive_samples, train_test_ratio)
    positive_samples_train_labels = np.ones(positive_samples_train.shape[0], dtype='uint8')
    positive_samples_test_labels = np.ones(positive_samples_test.shape[0], dtype='uint8')

    if verbose == 1:
        print ("Trovati", positive_samples.shape[0], "esempi positivi per il carattere", char.upper(), ".")
        print ("Campioni di training:", positive_samples_train.shape[0], "\tCampioni di test:", positive_samples_test.shape[0])

    # Calolo i samples negativi e le labels per il training e per il test
    neg_samples_n = int(positive_samples.shape[0]*pos_neg_ratio)
    negative_samples = samples_repository.get_n_negative_labeled_samples_by_char(char, neg_samples_n)
    negative_samples_train, negative_samples_test = split_by_ratio(negative_samples, train_test_ratio)
    negative_samples_train_labels = np.zeros(negative_samples_train.shape[0], dtype='uint8')
    negative_samples_test_labels = np.zeros(negative_samples_test.shape[0], dtype='uint8')

    if verbose == 1:
        print ("Richiesti", neg_samples_n, "esempi negativi: trovati", negative_samples.shape[0], "generici.")

    assert neg_samples_n == negative_samples.shape[0]

    train_imgs = np.append(positive_samples_train, negative_samples_train, axis=0)
    train_labels = np.append(positive_samples_train_labels, negative_samples_train_labels, axis=0)

    if verbose == 1:
        print ("Numero totale di campioni di training:", train_imgs.shape[0])

    test_imgs = np.append(positive_samples_test, negative_samples_test, axis=0)
    test_labels = np.append(positive_samples_test_labels, negative_samples_test_labels, axis=0)

    if verbose == 1:
        print ("Numero totale di campioni di test:", test_imgs.shape[0])

    return (train_imgs, train_labels, test_imgs, test_labels)


def generate_half_labeled_half_height(char, char_type, 
                                      pos_neg_ratio=pos_neg_ratio, train_test_ratio=train_test_ratio,
                                      neg_labeled_height_ratio=neg_labeled_height_ratio):

    # Calcolo i samples e le labels positivi per il training e il test
    positive_samples = samples_repository.get_all_positive_samples_by_char(char)
    positive_samples_train, positive_samples_test = split_by_ratio(positive_samples, train_test_ratio)
    positive_samples_train_labels = np.ones(positive_samples_train.shape[0], dtype='uint8')
    positive_samples_test_labels = np.ones(positive_samples_test.shape[0], dtype='uint8')

    print ("Trovati", positive_samples.shape[0], "esempi positivi per il carattere", char.upper(), ".")
    print ("Campioni di training:", positive_samples_train.shape[0], "\tCampioni di test:", positive_samples_test.shape[0])

    # Calolo i samples negativi e le labels per il training e per il test
    neg_samples_n = int(positive_samples.shape[0]*pos_neg_ratio)
    # divisi in generici e per altezza
    neg_labeled_samples_n, neg_height_samples_n = calc_ratio(neg_samples_n, neg_labeled_height_ratio)
    neg_labeled_samples = samples_repository.get_n_negative_labeled_samples_by_char(char, neg_labeled_samples_n)
    neg_height_samples = samples_repository.get_n_negative_samples_by_height_and_char(char, char_type, neg_height_samples_n)

    print ("Richiesti", neg_samples_n, "esempi negativi: trovati", neg_labeled_samples.shape[0], "generici e", neg_height_samples.shape[0], "per altezza.")

    assert neg_labeled_samples_n == neg_labeled_samples.shape[0]
    assert neg_height_samples_n == neg_height_samples.shape[0]

    neg_labeled_samples_train, neg_labeled_samples_test = split_by_ratio(neg_labeled_samples, train_test_ratio)
    neg_height_samples_train , neg_height_samples_test = split_by_ratio(neg_height_samples, train_test_ratio)

    print ("Samples negativi generici in training:", neg_labeled_samples_train.shape[0], "\tin test:", neg_labeled_samples_test.shape[0])
    print ("Samples negativi per altezza in training:", neg_height_samples_train.shape[0], "\tin test:", neg_height_samples_test.shape[0])

    neg_samples_train = np.append(neg_labeled_samples_train, neg_height_samples_train, axis=0)
    neg_samples_test = np.append(neg_labeled_samples_test, neg_height_samples_test, axis=0)
    neg_samples_train_labels = np.zeros(neg_samples_train.shape[0], dtype='uint8')
    neg_samples_test_labels = np.zeros(neg_samples_test.shape[0], dtype='uint8')

    print ("Campioni di training:", neg_samples_train.shape[0], "\tCampioni di test:", neg_samples_test.shape[0])

    train_imgs = np.append(positive_samples_train, neg_samples_train, axis=0)
    train_labels = np.append(positive_samples_train_labels, neg_samples_train_labels, axis=0)

    print ("Numero totale di campioni di training:", train_imgs.shape[0])

    test_imgs = np.append(positive_samples_test, neg_samples_test, axis=0)
    test_labels = np.append(positive_samples_test_labels, neg_samples_test_labels, axis=0)

    print ("Numero totale di campioni di test:", test_imgs.shape[0])

    return (train_imgs, train_labels, test_imgs, test_labels)

def generate_all_for_char_with_class(char, classification,
                                     train_test_ratio=train_test_ratio, verbose=1):

    # Calcolo i samples e le labels positivi per il training e il test
    positive_samples = samples_repository.get_all_positive_samples_by_char(char)
    positive_samples_train, positive_samples_test = split_by_ratio(positive_samples, train_test_ratio)
    positive_samples_train_labels = np.empty(positive_samples_train.shape[0], dtype='uint8')
    positive_samples_train_labels.fill(classification)
    positive_samples_test_labels = np.empty(positive_samples_test.shape[0], dtype='uint8')
    positive_samples_test_labels.fill(classification)

    if(verbose == 1):
        print ("Trovati", positive_samples.shape[0], "esempi positivi per il carattere", char.upper(), ".")
        print ("Campioni di training:", positive_samples_train.shape[0], "\tCampioni di test:", positive_samples_test.shape[0])

    train_imgs = positive_samples_train
    train_labels = positive_samples_train_labels # not actually labels, it's the class

    if(verbose == 1):
        print ("Numero totale di campioni di training:", train_imgs.shape[0])

    test_imgs = positive_samples_test
    test_labels = positive_samples_test_labels

    if(verbose == 1):
        print ("Numero totale di campioni di test:", test_imgs.shape[0])

    return (train_imgs, train_labels, test_imgs, test_labels)


def generate_all_chars_with_class(chars = ALPHABET_ALL,
                                  train_test_ratio=train_test_ratio, plot=False, verbose=1):
    
    sizes = np.zeros(len(chars))
    
    classifications = range(len(chars)) # this way chars[classification] = "a" if classification == 0
    
    (train_imgs, train_class, test_imgs, test_class) = generate_all_for_char_with_class(chars[0],\
                                                                                        classifications[0], \
                                                                                        verbose=verbose)
    sizes[0] = train_imgs.shape[0] + test_imgs.shape[0];
    
    for i in classifications:
        if (i>0):
            (train_imgs_prov, train_class_prov, test_imgs_prov, test_class_prov) = \
                                                            generate_all_for_char_with_class(chars[i],\
                                                            classifications[i], verbose=verbose)
                
            sizes[i] = train_imgs_prov.shape[0] + train_imgs_prov.shape[0]
            
            train_imgs = np.append(train_imgs, train_imgs_prov, axis=0)
            train_class = np.append(train_class, train_class_prov, axis=0)
            test_imgs = np.append(test_imgs, test_imgs_prov, axis=0)
            test_class = np.append(test_class, test_class_prov, axis=0)
            
    chars = np.asarray(chars)
            
    if plot:
        plt.plot(sizes, 'ro')
        plt.xticks(classifications, chars, rotation='vertical')
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.15)
        plt.show()

    return (train_imgs, train_class, test_imgs, test_class, chars)

def generate_all_chars_with_same_class(chars = ALPHABET_ALL, classification=0,
                                       train_test_ratio=train_test_ratio, plot=False, verbose=1):
    
    sizes = np.zeros(len(chars))
    
    classifications = range(len(chars)) # this way chars[classification] = "a" if classification == 0
    
    (train_imgs, train_class, test_imgs, test_class) = generate_all_for_char_with_class(chars[0],\
                                                                                        classification, \
                                                                                        verbose=verbose)
    sizes[0] = train_imgs.shape[0] + test_imgs.shape[0];
    
    for i in classifications:
        if (i>0):
            (train_imgs_prov, train_class_prov, test_imgs_prov, test_class_prov) = \
                                                            generate_all_for_char_with_class(chars[i],\
                                                            classification, verbose=verbose)
                
            sizes[i] = train_imgs_prov.shape[0] + train_imgs_prov.shape[0]
            
            train_imgs = np.append(train_imgs, train_imgs_prov, axis=0)
            train_class = np.append(train_class, train_class_prov, axis=0)
            test_imgs = np.append(test_imgs, test_imgs_prov, axis=0)
            test_class = np.append(test_class, test_class_prov, axis=0)
            
    chars = np.asarray(chars)
            
    if plot:
        plt.plot(sizes, 'ro')
        plt.xticks(classifications, chars, rotation='vertical')
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.15)
        plt.show()

    return (train_imgs, train_class, test_imgs, test_class, chars)

# Returns a dataset of bad cutted letters, useful for build a classificator that discriminate good cutted letters
# from bad cutted letters
def generate_bad_letters_of_chosen_chars(chars=ALPHABET_ALL, n_sample_for_class_width=100, split_ratio=0.7, verbose=0, plot=True):

    images = []

    yAxis = []

    for letter in chars:
        datas = samples_repository.get_n_negative_samples_by_width_and_char(letter, n_sample_for_class_width, verbose = verbose)
        yAxis.append(len(datas))
        images.extend(datas)

    images_len = len(images)
    split_value = int(images_len*split_ratio)
   
    indexes = [i for i in range(images_len)]
    np.random.shuffle(indexes)

    images = np.array(images)

    if plot:
        plt.plot(yAxis, 'ro')
        plt.xticks(np.arange(len(chars)),chars, rotation='vertical')
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.15)
        plt.show()

    return (images[indexes[:split_value]], images[indexes[split_value:]])

def generate_dataset_for_segmentator(verbose=0, plot=True, label_pos_class=1, label_neg_class=0):
    (X_train_Pos, y_train_Pos, X_test_Pos, y_test_Pos, _) = generate_all_chars_with_same_class(verbose=verbose, plot=plot, classification=label_pos_class)
    (X_train_Neg, X_test_Neg) = generate_bad_letters_of_chosen_chars(n_sample_for_class_width = 5000, plot=plot, verbose=verbose)
    
    X_train = []
    X_train.extend(X_train_Pos)
    X_train.extend(X_train_Neg)

    X_test = []
    X_test.extend(X_test_Pos)
    X_test.extend(X_test_Neg)

    y_train_Neg = [label_neg_class] * len(X_train_Neg)
    y_test_Neg = [label_neg_class] * len(X_test_Neg)

    y_train = []
    y_train.extend(y_train_Pos)
    y_train.extend(y_train_Neg)

    y_test = []
    y_test.extend(y_test_Pos)
    y_test.extend(y_test_Neg)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return (X_train, y_train, X_test, y_test)
