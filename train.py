import numpy as np
from skimage import feature
from skimage import transform
from skimage import util
from skimage import io
from skimage.color import rgb2gray

from my_functions import load_from_folder
from my_functions import resize_label
from my_functions import extract_rects
from my_functions import intersection_ratio
from my_functions import generate_random_negatives
from my_functions import find_faces
from my_functions import get_false_positives

from tqdm import tqdm

I_INDEX = 1
J_INDEX = 2
HEIGHT_INDEX = 3
WIDTH_INDEX = 4

print("Loading train labels...")
myLabels = np.loadtxt("label.txt", int)

print("Resizing labels to 1.5 ratio...")
myLabelsResized = np.zeros(myLabels.shape, int)
for i, label in enumerate(myLabels):
    myLabelsResized[i] = resize_label(label)

print("Loading train images...")
myImages = load_from_folder("train/")

print("Extracting faces from images...")
myFaces = np.zeros((len(myLabelsResized), 60, 40, 3))
i = 0
for image_index, image in enumerate(tqdm(myImages)):
    faces_for_image = extract_rects(image, myLabelsResized[np.where(myLabelsResized[:, 0] == image_index)])
    for false_positive in faces_for_image:
        myFaces[i] = transform.resize(false_positive, (60, 40), mode="constant", anti_aliasing=True)
        i += 1

print("Generating negative patches...")
random_per_image = 10
myNegatives = np.zeros((len(myImages) * random_per_image, 60, 40, 3))
for i in tqdm(range(len(myImages))):
    myNegatives[i * random_per_image:i * random_per_image + 10] = generate_random_negatives(random_per_image,
                                                                                            myImages[i],
                                                                                            myLabelsResized[np.where(
                                                                                                myLabelsResized[:,
                                                                                                0] == i)])

print("Computing negative HOGs...")
myNegativeHogs = np.zeros((len(myNegatives), 1215))
for i, negative in enumerate(tqdm(myNegatives)):
    myNegativeHogs[i] = (feature.hog(rgb2gray(util.img_as_float(negative)), block_norm='L2-Hys'))
myNegativeHogs = np.array(myNegativeHogs)

print("Computing positive HOGs...")
myPositiveHogs = np.zeros((len(myFaces), 1215))
for i, false_positive in enumerate(tqdm(myFaces)):
    myPositiveHogs[i] = feature.hog(rgb2gray(util.img_as_float(false_positive)), block_norm='L2-Hys')
myPositiveHogs = np.array(myPositiveHogs)

print("Concatenating and shuffling HOGs...")
myHogs = np.concatenate((myNegativeHogs, myPositiveHogs), axis=0)
myTargets = np.concatenate((np.full(myNegativeHogs.shape[0], -1.0), np.ones(myPositiveHogs.shape[0])))

# Generating permutation to shuffle HOGs
permutation = np.random.permutation(myHogs.shape[0])

myShuffledHogs = myHogs[permutation]
myShuffledTargets = myTargets[permutation]

# Learning
print("Learning...")

# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", gamma="scale")

from sklearn.svm import LinearSVC

clf = LinearSVC()

clf.max_iter = 3000
clf.fit(myShuffledHogs, myShuffledTargets)

from sklearn.model_selection import cross_validate

print("Cross validating classifier...")
cv_results = cross_validate(clf, myShuffledHogs, myShuffledTargets, cv=3, return_train_score=False)
print("Result of cross_validation :")
print(cv_results["test_score"])

# print("Finding faces on all images, this may take a while...")
# myDetections = []
# for i in tqdm(range(len(myImages))):
#     labels = find_faces(util.img_as_float(rgb2gray(myImages[i])), clf, i)
#     for label in labels:
#         myDetections.append(label)
# myDetections = np.array(myDetections)
# np.savetxt("detections_train.txt", myDetections)

# Loading from file if needed
myDetections = np.loadtxt("detections_train.txt").astype(int)

print("Extracting false positives labels from detections...")
myFalsePositivesLabels = []
for i in tqdm(range(myDetections[-1, 0] + 1)):
    detections = myDetections[np.where(myDetections[:, 0] == i)]
    faces = myLabelsResized[np.where(myLabelsResized[:, 0] == i)]
    myFalsePositivesLabels += get_false_positives(detections, faces)

myFalsePositivesLabels = np.array(myFalsePositivesLabels)

print("Extracting false positives from images...")
myFalsePositives = np.zeros((len(myFalsePositivesLabels), 60, 40, 3))
i = 0
for image_index, image in enumerate(tqdm(myImages)):
    false_positives_for_image = extract_rects(image, myFalsePositivesLabels[np.where(myFalsePositivesLabels[:, 0] == image_index)])
    for false_positive in false_positives_for_image:
        myFalsePositives[i] = transform.resize(false_positive, (60, 40), mode="constant", anti_aliasing=True)
        i += 1

print("Computing false positive HOGs...")
myFalsePositivesHogs = np.zeros((len(myFalsePositives), 1215))
for i, false_positive in enumerate(tqdm(myFalsePositives)):
    myFalsePositivesHogs[i] = (feature.hog(rgb2gray(util.img_as_float(false_positive)), block_norm='L2-Hys'))
myFalsePositivesHogs = np.array(myFalsePositivesHogs)

print("Concatenating and shuffling HOGs...")
myHogsWithFalsePositives = np.concatenate((myHogs, myFalsePositivesHogs), axis=0)
myTargetsWithFalsePositives = np.concatenate((myTargets, np.full(myFalsePositivesHogs.shape[0], -1.0)))

# Generating permutation to shuffle HOGs
permutation = np.random.permutation(myHogsWithFalsePositives.shape[0])

myShuffledHogsWithFalsePositives = myHogsWithFalsePositives[permutation]
myShuffledTargetsWithFalsePositives = myTargetsWithFalsePositives[permutation]

print("Learning with False Positives")
clf2 = LinearSVC()

clf2.max_iter = 3000
clf2.fit(myShuffledHogsWithFalsePositives, myShuffledTargetsWithFalsePositives)

print("Cross validating classifier...")
cv_results = cross_validate(clf2, myShuffledHogsWithFalsePositives, myShuffledTargetsWithFalsePositives, cv=3, return_train_score=False)
print("Result of cross_validation :")
print(cv_results["test_score"])

print("Finding faces on all images, this may take a while...")
myDetections = []
for i in tqdm(range(len(myImages))):
    labels = find_faces(util.img_as_float(rgb2gray(myImages[i])), clf2, i)
    for label in labels:
        myDetections.append(label)
myDetections = np.array(myDetections)
np.savetxt("detections_train.txt", myDetections)

from joblib import dump

print("Saving classifier to \"clf.joblib\"")
dump(clf, 'clf.joblib')

