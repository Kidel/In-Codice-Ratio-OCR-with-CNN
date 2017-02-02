from PIL import Image
import numpy as np

WIDTH=34
HEIGHT=56

def open_sample(filepath):
    """
    Parameters
    ----------
    filepath : str
        Path to an image file

    Returns
    -------
    list
        A list of lists which make it simple to access the greyscale value by
        im[y][x]
    """
    im = Image.open(filepath).convert('L')
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)
    greyscale_map = greyscale_map.reshape((height, width))
    return np.invert(add_white_padding(crop_white_space(greyscale_map)))

def open_many_samples(filenames):
    samples_images = []

    for filename in filenames:
        samples_images.append(open_sample(filename))

    return np.asarray(samples_images)

def crop_white_space(image, threshold=255):
    """
        rimuove lo spazio bianco eccedente, in alto, in basso, a dx e a sx.
        NB: funziona solo se l'immagine e' in grayscale
    """
    # una maschera di valori booleani. Ha la stessa struttura dell'immagine.
    # True se il pixel non e' bianco.
    img_mask = image < threshold
    # mask.any(1), mask.any(0) producono rispettivamente le maschere per righe e colonne:
    # True se la riga (o la colonna) contiene almeno un pixel nero.
    # sono monodimensionali.
    row_mask = img_mask.any(1)
    col_mask = img_mask.any(0)
    # np.ix_ costruisce gli indici che genereranno il prodotto fra le due maschere
    return image[np.ix_(row_mask, col_mask)]

def add_white_padding(img, width=WIDTH, height=HEIGHT):
    """
        Aggiunge un bordo bianco in alto e a destra fino a raggiungere
        le width e height desiderate
    """
    top = max(0, height)
    right = max(0, width)
    
    result = np.full((top, right), 255)

    result[result.shape[0]-img.shape[0]:result.shape[0],:img.shape[1]] = img
    
    return result