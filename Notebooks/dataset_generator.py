import samples_repository
import numpy as np

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


def generate_positive_and_negative_labeled(char, pos_neg_ratio=pos_neg_ratio, train_test_ratio=train_test_ratio):
    # Calcolo i samples e le labels positivi per il training e il test
    positive_samples = samples_repository.get_all_positive_samples_by_char(char)
    positive_samples_train, positive_samples_test = split_by_ratio(positive_samples, train_test_ratio)
    positive_samples_train_labels = np.ones(positive_samples_train.shape[0], dtype='uint8')
    positive_samples_test_labels = np.ones(positive_samples_test.shape[0], dtype='uint8')

    print ("Trovati", positive_samples.shape[0], "esempi positivi per il carattere", char.upper(), ".")
    print ("Campioni di training:", positive_samples_train.shape[0], "\tCampioni di test:", positive_samples_test.shape[0])

    # Calolo i samples negativi e le labels per il training e per il test
    neg_samples_n = int(positive_samples.shape[0]*pos_neg_ratio)
    negative_samples = samples_repository.get_n_negative_labeled_samples_by_char(char, neg_samples_n)
    negative_samples_train, negative_samples_test = split_by_ratio(negative_samples, train_test_ratio)
    negative_samples_train_labels = np.zeros(negative_samples_train.shape[0], dtype='uint8')
    negative_samples_test_labels = np.zeros(negative_samples_test.shape[0], dtype='uint8')

    print ("Richiesti", neg_samples_n, "esempi negativi: trovati", negative_samples.shape[0], "generici.")

    assert neg_samples_n == negative_samples.shape[0]

    train_imgs = np.append(positive_samples_train, negative_samples_train, axis=0)
    train_labels = np.append(positive_samples_train_labels, negative_samples_train_labels, axis=0)

    print ("Numero totale di campioni di training:", train_imgs.shape[0])

    test_imgs = np.append(positive_samples_test, negative_samples_test, axis=0)
    test_labels = np.append(positive_samples_test_labels, negative_samples_test_labels, axis=0)

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

