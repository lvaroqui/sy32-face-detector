from skimage import io
from skimage import transform
from skimage import util
from skimage import color
from skimage import feature
from skimage import data
from skimage import draw
from scipy.ndimage import filters
from skimage import util
import numpy as np
import matplotlib.pyplot as plt
import os


def load_from_folder(path):
    images = []
    files = os.listdir(path)
    files.sort()
    for f in files:
        images.append(io.imread(path + f))
    return images


myLabels = np.loadtxt("label.txt", int)

np.average(myLabels[:, 3] / myLabels[:, 4])


# 40*60


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


def draw_rectangle(img, face, color, thickness=3):
    img[face[1]:face[1] + face[3], face[2]:face[2] + thickness] = color
    img[face[1]:face[1] + face[3], face[2] + face[4]:face[2] + face[4] + thickness] = color
    img[face[1]:face[1] + thickness, face[2]:face[2] + face[4]] = color
    img[face[1] + face[3]:face[1] + face[3] + thickness, face[2]:face[2] + face[4] + thickness] = color
    return img


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


def extract_faces(images, labels):
    faces = []
    for i in range(0, len(images)):
        rows = np.where(labels[:, 0] == i)[0]
        for row in rows:
            label = labels[row]
            faces.append(crop_with_padding(images[i], label))
    return faces


myLabels = np.loadtxt("label.txt", int)
myLabelsResized = resize_label(myLabels)

myImages = load_from_folder("train/")

faces = extract_faces(myImages, myLabelsResized)

for face in faces:
    transform.resize(face, (60, 40))

io.imshow(faces[446])
io.show()

