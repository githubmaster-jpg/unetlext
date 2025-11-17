import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display
import torch

def plotter(data):
    fig , ax = plt.subplots(1,2, figsize=(14,5))
    ax[0].plot(data[0])
    ax[0].plot(data[1])
    ax[0].set_title('model loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'val'], loc='upper left')

    ax[1].plot(data[2])
    ax[1].set_title('model score')
    ax[1].set_ylabel('score')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['val'], loc='upper left')
    plt.show()  
    return None


# cannot get any predictions to work with unetlext not having predict, and code blocks avoiding that issue also causing errors
'''def eval(model, test_loader, device):
    show_predictions(model)
    return None

def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]

def show_predictions (model):
    sample_image = "/Users/tiia-/Downloads/cnn/Tibia/data/xrays/101.png"
    display([sample_image, create_mask(model.predict(sample_image[tf.newaxis, ...]))])
'''
