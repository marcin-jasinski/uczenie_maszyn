import os
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

from matplotlib import image
from matplotlib import pyplot

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint

from keras import backend as K
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from keras.models import Model
from keras.optimizers import Adam

from sklearn import metrics
import imblearn.metrics as imbmetrics

results = []
model_names = []


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='macro')


def g_mean_m(y_true, y_pred):
    return imbmetrics.geometric_mean_score(y_true, y_pred, average='macro')


def bac_m(y_true, y_pred):
    return metrics.balanced_accuracy_score(y_true, y_pred)

# custom activation function


def custom_activation(output):
    logexpsum = K.sum(K.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

# define the standalone supervised and unsupervised discriminator models


def define_discriminator(in_shape=(64, 64, 1), n_classes=2):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output layer nodes
    fe = Dense(n_classes)(fe)
    # supervised output
    c_out_layer = Activation('softmax')(fe)
    # define and compile supervised discriminator model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Adam(lr=0.0002, beta_1=0.5),
                    metrics=['accuracy', precision_m, recall_m])
    # unsupervised output
    d_out_layer = Lambda(custom_activation)(fe)
    # define and compile unsupervised discriminator model
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=0.0002, beta_1=0.5))
    return d_model, c_model

# define the standalone generator model


def define_generator(latent_dim):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 32x32 image
    n_nodes = 128 * 32 * 32
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((32, 32, 128))(gen)
    # upsample to 64x64
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (32, 32), activation='tanh', padding='same')(gen)
    # define model
    model = Model(in_lat, out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator


def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect image output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# load the images


def load_real_samples(inputDir):

    X = []
    y = []

    for filename in os.listdir(inputDir):
        img = image.imread(inputDir+filename)
        data = asarray(img)

        X.append(data)

        if "Normal" in filename:
            y.append(0)
        else:
            y.append(1)

    X = expand_dims(X, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5

    y = np.asarray(y)

    shuffle = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train_index, test_index in shuffle.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # print(X.shape, trainy.shape)
    return [X_train, np.asarray(y_train)], X_test, y_test

# select a supervised subset of the dataset, ensures classes are balanced


def select_supervised_samples(dataset, samples_per_class):
    X, y = dataset
    y = np.asarray(y, dtype=np.uint8)
    X_list, y_list = list(), list()

    for i in range(2):
        # get all images for this class
        X_with_class = X[y == i]
        # choose random instances
        ix = randint(0, len(X_with_class), samples_per_class[i])
        # add to list
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]

    return asarray(X_list), asarray(y_list)

# select real samples


def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

# use the generator to generate n fake examples, with class labels


def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return images, y

# generate samples and save as a plot and save the model


def summarize_performance(step, g_model, c_model, latent_dim, dataset, model_name, n_samples=16):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    # for i in range(16):
    #     # define subplot
    #     pyplot.subplot(4, 4, 1 + i)
    #     # turn off axis
    #     pyplot.axis('off')
    #     # plot raw pixel data
    #     pyplot.imshow(X[i, :, :, 0], cmap='gray_r')

    # save plot to file
    # filename1 = './generated_plots/generated_plot_%s.png' % (model_name)
    # pyplot.savefig(filename1)
    # pyplot.close()

    # evaluate the classifier model
    X, y = dataset
    _, acc, prec, recall = c_model.evaluate(X, y, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))

    # save the generator model
    # filename2 = 'g_model_%04d.h5' % (step+1)
    # g_model.save(filename2)

    # save the classifier model
    filename3 = 'models/c_model_%s.h5' % (model_name)
    c_model.save(filename3)
    print('>Saved: %s' % (model_name))

# train the generator and discriminator


def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, samples_per_class, model_name,
          n_epochs=50, n_batch=128):
    # select supervised dataset
    X_sup, y_sup = select_supervised_samples(dataset, samples_per_class)
    print(X_sup.shape, y_sup.shape)
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d'
          % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    # manually enumerate epochs
    for i in range(n_steps):
        # update supervised discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples(
            [X_sup, y_sup], half_batch)
        c_loss, c_acc, c_prec, c_recall = c_model.train_on_batch(
            Xsup_real, ysup_real)
        # update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(
            latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        # print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]'
        #       % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
        # evaluate the model performance
        if i == n_steps - 1:
            summarize_performance(i, g_model, c_model,
                                  latent_dim, dataset, model_name)


def evaluate_score(model, model_name, test_X, test_y):
    y_pred = model.predict(
        test_X)

    bac = bac_m(test_y, y_pred)
    f1_score = f1_m(test_y, y_pred)
    gmean = g_mean_m(test_y, y_pred)

    results.append([bac, f1_score, gmean])
    model_names.append(model_name)


# size of the latent space
latent_dim = 100

print("[INFO] Training SGAN on unbalanced dataset")

# load image data
ratios_unbalanced = [[13, 50], [25, 100], [50, 200], [100, 400]]

for ratio in ratios_unbalanced:
    print("[INFO] Samples ratio: %d" % ratio[0] + " : %d" % ratio[1])
    dataset, test_X, test_y = load_real_samples("./20_80/")
    K.clear_session()
    # create the discriminator models
    d_model, c_model = define_discriminator()
    # create the generator
    g_model = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    # model name
    model_name = "_unbalanced_%s_%s" % (ratio[0], ratio[1])
    # train model
    train(g_model, d_model, c_model, gan_model,
          dataset, latent_dim, ratio, model_name)
    # evaluate model
    evaluate_score(c_model, model_name, test_X, test_y)

print("[INFO] Training SGAN on balanced dataset")

# load image data
ratios_balanced = [[50, 50], [100, 100], [200, 200], [400, 400]]

# train model
for ratio in ratios_balanced:
    print("[INFO] Samples ratio: %d" % ratio[0] + " : %d" % ratio[1])
    dataset, test_X, test_y = load_real_samples("./50_50/")
    K.clear_session()
    # create the discriminator models
    d_model, c_model = define_discriminator()
    # create the generator
    g_model = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    # model name
    model_name = "_balanced_%s_%s" % (ratio[0], ratio[1])
    # train model
    train(g_model, d_model, c_model, gan_model,
          dataset, latent_dim, ratio, model_name)
    # evaluate model
    evaluate_score(c_model, model_name, test_X, test_y)

df = pd.DataFrame(results, columns=[
                  'BAC', 'F1-Score', 'G-Mean'], index=model_names)
df.to_csv('sgan_results.csv', index=True)
print(df)
