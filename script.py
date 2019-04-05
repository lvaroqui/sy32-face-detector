import os
from random import randint

import numpy as np
from skimage import feature
from skimage import io
from skimage import transform
from skimage import util
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

def load_from_folder(path):
    images = []
    files = os.listdir(path)
    files.sort()
    for f in files:
        images.append(io.imread(path + f))
    return images


myLabels = np.loadtxt("label.txt", int)

# 1
np.average(myLabels[:, 3] / myLabels[:, 4])


# 2 - Cropping image from train data
# On choisi une taille unique de 40*60

I_INDEX = 1
J_INDEX = 2
HEIGHT_INDEX = 3
WIDTH_INDEX = 4


def resize_label(lbls):
    labels = lbls.copy()
    # Resizing to ratio 1.5
    for face in labels:
        # If ratio smaller than 1.5
        if face[HEIGHT_INDEX] / face[WIDTH_INDEX] > 1.5:
            previous = face[WIDTH_INDEX]
            face[WIDTH_INDEX] = face[HEIGHT_INDEX] / 1.50
            face[J_INDEX] -= (face[WIDTH_INDEX] - previous) / 2
        # If ratio smaller than 1.5
        else:
            previous = face[HEIGHT_INDEX]
            face[HEIGHT_INDEX] = face[WIDTH_INDEX] * 1.50
            face[I_INDEX] -= (face[HEIGHT_INDEX] - previous) / 2
    return labels


def crop_with_padding(image, label):
    if label[1] < 0:
        image = util.pad(image, ((label[1] * -1, 0), (0, 0), (0, 0)), 'constant')
        label[1] = 0
    if label[2] < 0:
        image = util.pad(image, ((0, 0), (label[2] * -1, 0), (0, 0)), 'constant')
        label[2] = 0
    if label[1] + label[3] > image.shape[0]:
        image = util.pad(image, ((image.shape[0] + label[1] + label[3], 0), (0, 0), (0, 0)), 'constant')
    if label[1] < 0:
        image = util.pad(image, ((0, 0), (0, image.shape[0] + label[2] + label[4]), (0, 0)), 'constant')

    return image[label[1]:label[1] + label[3], label[2]:label[2] + label[4]]


def get_labels_from_image_index(index, labels):
    return labels[np.where(labels[:, 0] == index)]


def get_faces_from_image_index(index, faces, labels):
    return faces[np.where(labels[:, 0] == index)]


def extract_faces(images, labels):
    faces = []
    for i in range(0, len(images)):
        image_labels = get_labels_from_image_index(i, labels)
        for label in image_labels:
            faces.append(crop_with_padding(images[i], label))
    return faces


myLabels = np.loadtxt("label.txt", int)
myLabelsResized = resize_label(myLabels)

myImages = load_from_folder("train/")

faces = extract_faces(myImages, myLabelsResized)

for i in range(len(faces)):
    faces[i] = transform.resize(faces[i], (60, 40), mode="constant")

faces = np.array(faces)


# 3 - Generating random no match pattern

# Return the ratio of intersection between two rectangle
# label format : [id, i, j, height, width]
def intersec(rect1, rect2):
    aire_rect1 = rect1[HEIGHT_INDEX] * rect1[WIDTH_INDEX]
    aire_rect2 = rect2[HEIGHT_INDEX] * rect2[WIDTH_INDEX]
    intersec = 0

    # Check if the rectangles overlap
    if (not (
            rect1[J_INDEX] + rect1[WIDTH_INDEX] < rect2[J_INDEX] or rect1[J_INDEX] > rect2[J_INDEX] + rect2[WIDTH_INDEX] or rect1[I_INDEX] + rect1[HEIGHT_INDEX] < rect2[I_INDEX] or
            rect1[I_INDEX] > rect2[I_INDEX] + rect2[HEIGHT_INDEX])):
        top = max(rect1[I_INDEX], rect2[I_INDEX])
        bottom = min(rect1[I_INDEX] + rect1[HEIGHT_INDEX], rect2[I_INDEX] + rect2[HEIGHT_INDEX])
        left = max(rect1[J_INDEX], rect2[J_INDEX])
        right = min(rect1[J_INDEX] + rect1[WIDTH_INDEX], rect2[J_INDEX] + rect2[WIDTH_INDEX])
        intersec = (bottom - top) * (right - left)

    return intersec / (aire_rect2 + aire_rect1 - intersec)


# Generate random not face patches of size (40*60) from an image
# number : number to be generated
# image : image from witch to extract patches
# labels : labels describing face on the image
def generate_random_not_faces(number, image, labels):
    result_labels = []
    while len(result_labels) < number:
        new_label = [0, 0, 0, 0, 0]
        is_good = True

        new_label[HEIGHT_INDEX] = randint(20, image.shape[0])
        new_label[I_INDEX] = randint(0, image.shape[0] - new_label[HEIGHT_INDEX])
        new_label[WIDTH_INDEX] = int(new_label[HEIGHT_INDEX] / 1.5)
        if new_label[WIDTH_INDEX] < image.shape[1]:
            new_label[J_INDEX] = randint(0, image.shape[1] - new_label[WIDTH_INDEX])
        else:
            continue

        for label in labels:
            if intersec(label, new_label) > 0.50:
                is_good = False
                continue

        if is_good:
            result_labels.append(new_label)

    result = []
    for label in result_labels:
        zone = image[label[I_INDEX]:label[I_INDEX] + label[HEIGHT_INDEX], label[J_INDEX]:label[J_INDEX] + label[WIDTH_INDEX]]
        result.append(transform.resize(zone, (60, 40), mode="constant"))
    return result


not_faces = []
for i in range(len(myImages)):
    not_faces += generate_random_not_faces(10, myImages[i], get_labels_from_image_index(i, myLabels))
not_faces = np.array(not_faces)


# Q4 - Apprendre un classifieur

# HOG Computation
hogs_not_faces = []
for not_face in not_faces:
    hogs_not_faces.append(feature.hog(rgb2gray(util.img_as_float(not_face))))
hogs_not_faces = np.array(hogs_not_faces)

hogs_faces = []
for face in faces:
    hogs_faces.append(feature.hog(rgb2gray(util.img_as_float(face))))
hogs_faces = np.array(hogs_faces)

hogs = np.concatenate((hogs_not_faces, hogs_faces), axis=0)
targets = np.concatenate((np.full(hogs_not_faces.shape[0], -1.0), np.ones(hogs_faces.shape[0])))

permutation = np.random.permutation(hogs.shape[0])

hogs_shuffled = hogs[permutation]
targets_shuffled = targets[permutation]

# Learning
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
clf = LinearSVC()
clf.max_iter = 3000

cv_results = cross_validate(clf, hogs_shuffled, targets_shuffled, cv=6, return_train_score=False)
print(cv_results["test_score"])

