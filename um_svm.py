import os
import cv2
import math
import operator
import numpy as np
import pandas as pd

from matplotlib import image
from numpy.random import randint

# GLCM libs
from skimage.feature import greycoprops
from skimage.feature import greycomatrix

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
import imblearn.metrics as imbmetrics

results = []
model_names = []


def f1_m(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='macro')


def g_mean_m(y_true, y_pred):
    return imbmetrics.geometric_mean_score(y_true, y_pred, average='macro')


def bac_m(y_true, y_pred):
    return metrics.balanced_accuracy_score(y_true, y_pred)


def load_real_samples(inputDir):
    X = []
    y = []

    # GLCM distances & angles (in radians)
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, np.pi * 0.75]
    props = ["contrast", "dissimilarity",
             "homogeneity", "ASM", "energy", "correlation"]

    print("[INFO] Processing images")

    for filename in os.listdir(inputDir):
        img = cv2.imread(inputDir+filename)
        # Crop image to square (center-based)
        crop_dim = min(img.shape[0], img.shape[1])
        bounding = (crop_dim, crop_dim)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        crop_img = img[slices]

        target_size = 64

        # INTER_CUBIC interpolation for enlarging images, INTER_AREA for shrinking
        interpolation = cv2.INTER_CUBIC if crop_img.shape[0] <= target_size else cv2.INTER_AREA
        dsize = (target_size, target_size)
        reshaped = cv2.resize(crop_img, dsize, interpolation)

        # Convert RGB to grayscale
        grayscale = cv2.cvtColor(reshaped, cv2.COLOR_BGR2GRAY)
        img_array = np.asarray(grayscale, dtype=np.uint8)

        # GLCM
        g_matrix = greycomatrix(img_array, distances,
                                angles, normed=True, symmetric=True)
        img_features = np.ravel(
            [np.ravel(greycoprops(g_matrix, prop)) for prop in props]).T

        X.append(img_features)

        if "Normal" in filename:
            y.append(0)
        elif "Pneumonia-Bacterial" in filename:
            y.append(1)

    X = np.asarray(X)
    X = (X - 127.5) / 127.5

    y = np.asarray(y)

    shuffle = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train_index, test_index in shuffle.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # print(X.shape, trainy.shape)
    return [X_train, y_train.astype('float32')], X_test, y_test.astype('float32')


def select_supervised_samples(dataset, samples_per_class):
    X, y = dataset
    y = np.asarray(y, dtype=np.uint8)
    unlabeled_ix = {n for n in range(len(y))}
    X_list, y_list, U_list = list(), list(), list()

    for i in range(2):
        # get all images for this class
        X_with_class = X[y == i]
        # choose random instances
        ix = randint(0, len(X_with_class), samples_per_class[i])
        unlabeled_ix = unlabeled_ix.difference(set(ix))
        # add to list
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]

    [U_list.append(X[i]) for i in unlabeled_ix]

    return np.asarray(X_list), np.asarray(y_list), np.asarray(U_list)


print("[INFO] Reading unbalanced dataset")
dataset, X_test, y_test = load_real_samples("./20_80/")

print("[INFO] Training SVM on 20-80 dataset \n")

ratios_unbalanced = [[13, 50], [25, 100], [50, 200], [100, 400]]

for ratio in ratios_unbalanced:
    model_name = "_unbalanced_%s_%s" % (ratio[0], ratio[1])
    print("[INFO] " + model_name)
    clf = SVC(kernel="rbf", gamma=0.01, C=1000)
    X_sup, y_sup, unlabeled_samples = select_supervised_samples(dataset, ratio)

    X_train, y_train = X_sup, y_sup
    while unlabeled_samples.shape != (0, 48):
        clf.fit(X_train, y_train)
        samp, unlabeled_samples = unlabeled_samples[-1].reshape(
            1, -1), unlabeled_samples[:-1]
        y_pred = clf.predict(samp)
        X_train = np.append(X_train, samp, axis=0)
        y_train = np.append(y_train, y_pred)
        if unlabeled_samples.shape[0] % 100 == 0:
            print("[INFO] Unlabeled set size: %d" %
                  (unlabeled_samples.shape[0]))

    y_pred = clf.predict(X_test)

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    bac = bac_m(y_test, y_pred)
    f1_score = f1_m(y_test, y_pred)
    gmean = g_mean_m(y_test, y_pred)

    results.append([bac, f1_score, gmean])
    model_names.append(model_name)

print("[INFO] Reading balanced dataset")
dataset, X_test, y_test = load_real_samples("./50_50/")

print("[INFO] Training SVM on 50-50 dataset \n")

ratios_balanced = [[50, 50], [100, 100], [200, 200], [400, 400]]

for ratio in ratios_balanced:
    model_name = "_balanced_%s_%s" % (ratio[0], ratio[1])
    print("[INFO] " + model_name)
    clf = SVC(kernel="rbf", gamma=0.01, C=1000)
    X_sup, y_sup, unlabeled_samples = select_supervised_samples(dataset, ratio)

    X_train, y_train = X_sup, y_sup
    while unlabeled_samples.shape != (0, 48):
        clf.fit(X_train, y_train)
        samp, unlabeled_samples = unlabeled_samples[-1].reshape(
            1, -1), unlabeled_samples[:-1]
        y_pred = clf.predict(samp)
        X_train = np.append(X_train, samp, axis=0)
        y_train = np.append(y_train, y_pred)
        if unlabeled_samples.shape[0] % 100 == 0:
            print("[INFO] Unlabeled set size: %d" %
                  (unlabeled_samples.shape[0]))

    y_pred = clf.predict(X_test).astype('float32')

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    bac = bac_m(y_test, y_pred)
    f1_score = f1_m(y_test, y_pred)
    gmean = g_mean_m(y_test, y_pred)

    results.append([bac, f1_score, gmean])
    model_names.append(model_name)

df = pd.DataFrame(results, columns=[
                  'BAC', 'F1-Score', 'G-Mean'], index=model_names)
df.to_csv('svm_results.csv', index=True)
print(df)
