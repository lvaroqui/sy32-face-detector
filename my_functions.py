"""
Custom functions for face detector project
"""
import os
import time
from random import randint
from skimage.color import rgb2gray

import numpy as np
from skimage import feature
from skimage import io
from skimage import transform
from skimage import util
from tqdm import tqdm
from matplotlib import pyplot as plt

from multiprocessing import Pool

I_INDEX = 1
J_INDEX = 2
HEIGHT_INDEX = 3
WIDTH_INDEX = 4


def load_from_folder(path):
    """
    Return a list of all images from the given path
    :param path: path from which to
    :return: list of loaded images
    """
    images = []
    files = os.listdir(path)
    files.sort()
    for file in tqdm(files):
        images.append(io.imread(path + file))
    return images


def get_label_with_index(labels, index):
    return labels[np.where(labels[:, 0] == index)]


def resize_label(label):
    """
    Resize a given label to a given HEIGHT / WIDTH ratio
    :param label: label to be resized
    :param ratio: ratio Height / Width
    :return: resized label
    """
    # If current ratio greater than given ratio
    if label[HEIGHT_INDEX] / label[WIDTH_INDEX] > 1.5:
        previous = label[WIDTH_INDEX]
        label[WIDTH_INDEX] = label[HEIGHT_INDEX] / 1.5
        label[J_INDEX] -= (label[WIDTH_INDEX] - previous) / 2
    # If current ratio smaller than given ratio
    else:
        previous = label[HEIGHT_INDEX]
        label[HEIGHT_INDEX] = label[WIDTH_INDEX] * 1.5
        label[I_INDEX] -= (label[HEIGHT_INDEX] - previous) / 2
    return label


def _extract_rects_with_padding(image, label):
    """
    Extract a rect from an image according to the given label
    The function pads the cropped image with black if the rect oversteps the image
    :param image: image from which extract label
    :param label: label to be extracted
    :return: padded extracted rect
    """
    if label[I_INDEX] < 0:
        image = util.pad(image, ((label[I_INDEX] * -1, 0), (0, 0), (0, 0)), 'constant')
        label[I_INDEX] = 0
    if label[J_INDEX] < 0:
        image = util.pad(image, ((0, 0), (label[2] * -1, 0), (0, 0)), 'constant')
        label[J_INDEX] = 0
    if label[I_INDEX] + label[HEIGHT_INDEX] > image.shape[0]:
        image = util.pad(image, ((image.shape[0] + label[I_INDEX] + label[HEIGHT_INDEX], 0), (0, 0), (0, 0)),
                         'constant')
    if label[I_INDEX] < 0:
        image = util.pad(image, ((0, 0), (0, image.shape[0] + label[J_INDEX] + label[WIDTH_INDEX]), (0, 0)), 'constant')

    return image[label[I_INDEX]:label[I_INDEX] + label[HEIGHT_INDEX],
                 label[J_INDEX]:label[J_INDEX] + label[WIDTH_INDEX]]


def extract_rects(image, labels):
    """
    Extract rects associated with n labels from image
    :param image: image from which extract labels
    :param labels: labels to be extracted
    :return: extracted rects
    """
    faces = []
    for label in labels:
        faces.append(_extract_rects_with_padding(image, label))
    return faces


def intersection_ratio(label1, label2):
    """
    Compute intersection ratio of two labels
    :param label1: First label
    :param label2: Second Label
    :return: ratio of intersection between two rectangle
    """
    aire_rect1 = label1[HEIGHT_INDEX] * label1[WIDTH_INDEX]
    aire_rect2 = label2[HEIGHT_INDEX] * label2[WIDTH_INDEX]
    result = 0

    # Check if the rectangles overlap
    if (not (label1[J_INDEX] + label1[WIDTH_INDEX] < label2[J_INDEX] or label1[J_INDEX] > label2[J_INDEX] + label2[
             WIDTH_INDEX] or label1[I_INDEX] + label1[HEIGHT_INDEX] < label2[I_INDEX] or
             label1[I_INDEX] > label2[I_INDEX] + label2[HEIGHT_INDEX])):
        top = max(label1[I_INDEX], label2[I_INDEX])
        bottom = min(label1[I_INDEX] + label1[HEIGHT_INDEX], label2[I_INDEX] + label2[HEIGHT_INDEX])
        left = max(label1[J_INDEX], label2[J_INDEX])
        right = min(label1[J_INDEX] + label1[WIDTH_INDEX], label2[J_INDEX] + label2[WIDTH_INDEX])
        result = (bottom - top) * (right - left)

    return result / (aire_rect2 + aire_rect1 - result)


def generate_random_negatives(number, image, labels):
    """
    Generate random not face patches of size (60*40) from an image
    :param number: number of negative to be generated
    :param image: image from witch to extract patches
    :param labels: labels describing faces on the image
    :return: random negatives labels
    """
    result_labels = np.zeros((10, 5), int)
    i = 0
    while i < number:
        is_good = True

        result_labels[i, HEIGHT_INDEX] = randint(20, image.shape[0])
        result_labels[i, I_INDEX] = randint(0, image.shape[0] - result_labels[i, HEIGHT_INDEX])
        result_labels[i, WIDTH_INDEX] = int(result_labels[i, HEIGHT_INDEX] / 1.5)
        if result_labels[i, WIDTH_INDEX] < image.shape[1]:
            result_labels[i, J_INDEX] = randint(0, image.shape[1] - result_labels[i, WIDTH_INDEX])
        else:
            continue

        for label in labels:
            if intersection_ratio(label, result_labels[i]) > 0.50:
                is_good = False
                continue

        if is_good:
            i += 1

    result = np.zeros((10, 60, 40, 3))
    for i, label in enumerate(result_labels):
        zone = image[label[I_INDEX]:label[I_INDEX] + label[HEIGHT_INDEX],
               label[J_INDEX]:label[J_INDEX] + label[WIDTH_INDEX]]
        result[i] = transform.resize(zone, (60, 40), mode="constant", anti_aliasing=True)

    return result


def _remove_duplicates(labels):
    """
    Remove duplicates in labels that have the same index
    :param labels: labels from which remove duplicates
    :return: labels without duplicates
    """
    to_delete = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if intersection_ratio(labels[i], labels[j]) >= 0.5:
                to_delete.append(j)
    to_delete = np.unique(np.array(to_delete))
    return np.delete(labels, to_delete, 0)


def find_faces(image, clf, index, vstep=15, hstep=15, dmax=2.5, dstep=0.5):
    """
    Find faces in given image with a given classifier with sliding window
    :param image: image in which to find faces
    :param clf: classifier to recognize face
    :param index: index of the image
    :param vstep: vertical step taken during sliding window
    :param hstep: horizontal step taken during sliding window
    :param dmax: maximum divider for image during sliding window
    :param dstep: divider step during sliding window
    :return: labels with identified faces
    """
    img_height, img_width = image.shape[0], image.shape[1]

    labels = []
    for divider in np.arange(1, dmax, dstep):
        if divider > 1:
            new_width, new_height = int(img_width / divider), int(img_height / divider)
            image = transform.resize(image, (new_height, new_width), anti_aliasing=True)
        else:
            new_width, new_height = img_width, img_height

        for i in range(0, new_height - 60, vstep):
            for j in range(0, new_width - 40, hstep):
                face = image[i:i + 60, j:j + 40]
                hog = feature.hog(face)
                if clf.predict([hog]) == 1:
                    labels.append([index, i*divider, j*divider, 60 * divider, 40 * divider])

    return _remove_duplicates(np.array(labels))


def scan_images(images, clf, first_index, vstep=15, hstep=15, dmax=2.5, dstep=0.5):
    """
    Scan image synchronously

    :param images: images to scan
    :param clf: classifier
    :param first_index: first_index of scanned image
    :param vstep: vertical step taken during sliding window
    :param hstep: horizontal step taken during sliding window
    :param dmax: maximum divider for image during sliding window
    :param dstep: divider step during sliding window
    :return:
    """
    detections = []
    for i in tqdm(range(len(images))):
        labels = find_faces(util.img_as_float(rgb2gray(images[i])), clf, first_index + i,
                            vstep, hstep, dmax, dstep)
        for label in labels:
            detections.append(label)
    return np.array(detections)


def scan_images_multiprocessed(images, clf, processes, vstep=15, hstep=15, dmax=2.5, dstep=0.5):
    """
    Scan a batch of images with multiple process
    :param images: images to scan
    :param clf: classifier to scan images with
    :param processes: number of process to run
    :param vstep: vertical step taken during sliding window
    :param hstep: horizontal step taken during sliding window
    :param dmax: maximum divider for image during sliding window
    :param dstep: divider step during sliding window
    :return: face labels
    """
    pool = Pool(processes=processes)  # start 4 worker processes
    results = []
    for i in range(0, processes):
        begin = i*int(len(images)/processes)
        if i == processes - 1:
            end = len(images)
        else:
            end = (i+1) * int(len(images) / processes)
        results.append(pool.apply_async(scan_images, (images[begin:end], clf, begin, vstep, hstep, dmax, dstep)))
    detections = []
    for result in results:
        detections.append(result.get())
    return np.concatenate(detections).astype(int)


def get_false_positives(detections, faces):
    """
    Return false positive labels
    :param detections: detections from which you want to find false positives
    :param faces: actual faces
    :return: false positive labels
    """
    false_positives = []
    for detection in detections:
        is_positive = False
        for face in faces:
            if intersection_ratio(detection, face) > 0.5:
                is_positive = True
                break
        if not(is_positive):
            false_positives.append(detection)

    return false_positives


def stats(detections, faces):
    """
    Return the stats for a set of detections
    :param detections: detections
    :param faces: actual faces
    :return: precisions, rappel, F-score
    """
    vp, fp, fn, vn = 0, 0, 0, 0
    max_label = np.max(faces[:, 0])
    for i in range(max_label+1):
        detections_i = get_label_with_index(detections, i)
        faces_i = get_label_with_index(faces, i)
        local_vp = 0
        for face in faces_i:
            found = False
            for detection in detections_i:
                if intersection_ratio(face, detection) >= 0.5:
                    found = True
                    break
            if found:
                vp += 1
                local_vp += 1
            else:
                fn += 1
        fp += len(detections_i) - local_vp

    precision = vp / (vp + fp)
    rappel = vp / (vp + fn)
    f_score = 2*((precision*rappel)/(precision+rappel))

    return precision, rappel, f_score