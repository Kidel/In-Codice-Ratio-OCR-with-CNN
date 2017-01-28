import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
import os.path

class Util(object):
    
    def plot_image(self, image, img_shape=(28,28)):
        plt.imshow(image.reshape(img_shape),
                   interpolation='nearest',
                   cmap='binary')

        plt.show()
    
    def plot_images(self, images, cls_true, cls_pred=None, img_size=28, img_shape=(28,28), interpolation='none'):
        assert len(images) == len(cls_true) == 9
        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i].reshape(img_shape), cmap='binary', interpolation=interpolation)
            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "Class: {0}".format(cls_true[i])
            else:
                xlabel = "Class: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def maybe_save_network(self, model, model_image_path='images/temp.png'):
        if not os.path.exists(model_image_path):
            plot(model, to_file=model_image_path)
        return model_image_path