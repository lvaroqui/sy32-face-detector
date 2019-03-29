from skimage import io
from skimage import transform
from skimage import util
from skimage import color
from skimage import feature
from skimage import data
from skimage import draw
from scipy.ndimage import filters
from random import randint
from skimage import util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


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


# 2 - Cropping image from train datas
# On choisi une taille unique de 40*60


def resize_label(lbls):
    labels = lbls.copy()
    # Resizing to ratio 1.5
    for face in labels:
        # If ratio smaller than 1.5
        if face[3] / face[4] > 1.5:
            previous = face[4]
            face[4] = face[3] / 1.50
            face[2] -= (face[4] - previous) / 2
        # If ratio smaller than 1.5
        else:
            previous = face[3]
            face[3] = face[4] * 1.50
            face[1] -= (face[3] - previous) / 2
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
    return np.array(faces)


myLabels = np.loadtxt("label.txt", int)
myLabelsResized = resize_label(myLabels)

myImages = load_from_folder("train/")

faces = extract_faces(myImages, myLabelsResized)

for i in range(len(faces)):
    faces[i] = transform.resize(faces[i], (60, 40), mode="constant")


# 3 - Generating random no match pattern

# Return the ration of intersection between two rectangle
# label format : [id, i, j, height, width]
def intersec(rect1, rect2):
    aire_rect1 = rect1[3] * rect1[4]
    aire_rect2 = rect2[3] * rect2[4]
    intersec = 0

    # Check if the rectangles overlap
    if (not (
            rect1[2] + rect1[4] < rect2[2] or rect1[2] > rect2[2] + rect2[4] or rect1[1] + rect1[3] < rect2[1] or
            rect1[1] > rect2[1] + rect2[3])):
        top = max(rect1[1], rect2[1])
        bottom = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
        left = max(rect1[2], rect2[2])
        right = min(rect1[2] + rect1[4], rect2[2] + rect2[4])
        intersec = (bottom - top) * (right - left)

    return 2 * intersec / (aire_rect2 + aire_rect1)


# Generate random not face patches of size (40*60) from an image
# number : number to be generated
# image : image from witch to extract patches
# labels : labels describing face on the image
def generate_random_not_faces(number, image, labels):
    result_labels = []
    while len(result_labels) != number:
        new_label = [0, 0, 0, 0, 0]
        new_label[3] = randint(20, image.shape[0])
        new_label[1] = randint(0, image.shape[0] - new_label[3])
        new_label[4] = int(new_label[3] / 1.5)
        new_label[2] = randint(0, image.shape[1] - new_label[4])

        is_good = True
        for label in labels:
            if intersec(label, new_label) > 0.10:
                is_good = False
                break

        if is_good:
            result_labels.append(new_label)

    result = []
    for label in result_labels:
        zone = image[label[1]:label[1] + label[3], label[2]:label[2] + label[4]]
        result.append(transform.resize(zone, (60, 40), mode="constant"))

    return result


rnds = generate_random_not_faces(10, myImages[0], get_labels_from_image_index(0, myLabels))


# fig, ax = plt.subplots(1)
# rect = Rectangle((rnd[2], rnd[1]), rnd[4], rnd[3])
# ax.add_patch(rect)
# plt.show()
