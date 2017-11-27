from keras.callbacks import Callback
import numpy as np
from PIL import Image
from deepsense import neptune
from glob import glob
from matplotlib import pyplot as plt
from scipy.misc import imresize

def load_Xy(path):
    filenames0 = glob(path + "cat.*.jpg")
    filenames1 = glob(path + "dog.*.jpg")
    filenames0_valid = filenames0[0:1000]
    filenames0_train = filenames0[1000:]
    filenames1_valid = filenames1[0:1000]
    filenames1_train = filenames1[1000:]
    X_train = np.zeros((len(filenames0_train) + len(filenames1_train), 256, 256, 3))
    X_valid = np.zeros((len(filenames0_valid) + len(filenames1_valid), 256, 256, 3))
    X_train = X_train.astype('uint8')
    X_valid = X_valid.astype('uint8')
    y_train = np.zeros(len(filenames0_train) + len(filenames1_train))
    y_valid = np.zeros(len(filenames0_valid) + len(filenames1_valid))
    y_train[len(filenames0_train):] = 1.
    y_valid[len(filenames0_valid):] = 1.
                                                                
    size = (256, 256, 3)

    print(path + "cat.*.png" + " valid")
    for i, filename in enumerate(filenames0_valid):
        im = Image.open(filename)
        im = imresize(im, size)
        X_valid[i] = im
        if i % 100 == 0:
            print(i, end=" ")

    print("\n")
    print(path + "dogs/*.png" + " valid")
    for i, filename in enumerate(filenames1_valid):
        im = Image.open(filename)
        im = imresize(im, size)
        X_valid[len(filenames0_valid) + i] = im
        if i % 100 == 0:
            #plots images to check if they are correctly loaded
            #plt.imshow(X_valid[len(filenames0_valid) + i])
            #plt.show()
            print(i, end=" ")  
                                                                                                                                                                                                             
    print("\n") 
    print(path + "cat.*.png" + " train")
    for i, filename in enumerate(filenames0_train):
        im = Image.open(filename)
        im = imresize(im, size)
        X_train[i] = im
        if i % 100 == 0:
            print(i, end=" ")

    print("\n")
    print(path + "dogs/*.png" + " train")
    for i, filename in enumerate(filenames1_train):
        im = Image.open(filename)
        im = imresize(im, size)
        X_train[len(filenames0_train) + i] = im
        if i % 100 == 0:
            print(i, end=" ")
    
    return X_train, y_train, X_valid, y_valid

ctx = neptune.Context()

def array_2d_to_image(array, autorescale=True):
    assert array.min() >= 0
    assert len(array.shape) in [2, 3]
    if array.max() <= 1 and autorescale:
        array = 255 * array
    array = array.astype('uint8')
    return Image.fromarray(array)

def model_summary(model):
    print("Model created successfully.")
    print(model.summary())
    ctx.channel_send('n_layers', len(model.layers))
    ctx.channel_send('n_parameters', model.count_params())

categories = ['cat', 'dog']

class NeptuneCallback(Callback):
    def __init__(self, x_test, y_test, images_per_epoch=-1):

        try:
            ctx.channel_reset('Log-loss training')
            ctx.channel_reset('Log-loss validation')
            ctx.channel_reset('Accuracy training')
            ctx.channel_reset('Accuracy validation')
            ctx.channel_reset('false_predictions')
        except:
            pass
        self.epoch_id = 0
        self.images_per_epoch = images_per_epoch
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # logging numeric channels
        ctx.channel_send('Log-loss training', self.epoch_id, logs['loss'])
        ctx.channel_send('Log-loss validation', self.epoch_id, logs['val_loss'])
        ctx.channel_send('Accuracy training', self.epoch_id, logs['acc'])
        ctx.channel_send('Accuracy validation', self.epoch_id, logs['val_acc'])

        # Predict the digits for images of the test set.
        validation_predictions = self.model.predict_classes(self.x_test)
        scores = self.model.predict(self.x_test)

        # Identify the incorrectly classified images and send them to Neptune Dashboard.
        image_per_epoch = 0
        for index, (prediction, actual) in enumerate(zip(validation_predictions, self.y_test.argmax(axis=1))):
            if prediction != actual:
                if image_per_epoch == self.images_per_epoch:
                    break
                image_per_epoch += 1

                ctx.channel_send('false_predictions', neptune.Image(
                    name='[{}] {} X {} V'.format(self.epoch_id, categories[prediction], categories[actual]),
                    description="\n".join([
                        "{:5.1f}% {} {}".format(100 * score, categories[i], "!!!" if i == actual else "")
                        for i, score in enumerate(scores[index])]),
                    data=array_2d_to_image(self.x_test[index,:,:])))
