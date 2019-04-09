import numpy as np
from skimage import feature
from skimage import transform
from skimage import util
from skimage.color import rgb2gray

from my_functions import load_from_folder
from my_functions import resize_label
from my_functions import extract_rects
from my_functions import intersection_ratio
from my_functions import generate_random_negatives
from my_functions import find_faces
from my_functions import remove_duplicates

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
    for face in faces_for_image:
        myFaces[i] = transform.resize(face, (60, 40), mode="constant", anti_aliasing=True)
        i += 1

print("Generating negative patches...")
random_per_image = 10
myNegatives = np.zeros((len(myImages)*random_per_image, 60, 40, 3))
for i in tqdm(range(len(myImages))):
    myNegatives[i * random_per_image:i*random_per_image + 10] = generate_random_negatives(random_per_image, myImages[i], myLabelsResized[np.where(myLabelsResized[:, 0] == i)])

print("Computing negative HOGs...")
myNegativeHogs = np.zeros((len(myNegatives), 1215))
for i, negative in enumerate(tqdm(myNegatives)):
    myNegativeHogs[i] = (feature.hog(rgb2gray(util.img_as_float(negative)), block_norm='L2-Hys'))
myNegativeHogs = np.array(myNegativeHogs)

print("Computing positive HOGs...")
myPositiveHogs = np.zeros((len(myFaces), 1215))
for i, face in enumerate(tqdm(myFaces)):
    myPositiveHogs[i] = feature.hog(rgb2gray(util.img_as_float(face)), block_norm='L2-Hys')
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

from sklearn.model_selection import cross_validate

# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", gamma="scale")

from sklearn.svm import LinearSVC
clf = LinearSVC()


clf.max_iter = 3000
clf.fit(myShuffledHogs, myShuffledTargets)

print("Cross validating classifier...")
cv_results = cross_validate(clf, myShuffledHogs, myShuffledTargets, cv=3, return_train_score=False)
print("Result of cross_validation :")
print(cv_results["test_score"])

# print("Finding faces on all images, this may take a while...")
# myDetectedFaces = []
# for i in tqdm(range(len(myImages))):
#     labels = find_faces(util.img_as_float(rgb2gray(myImages[i])), clf, i)
#     for label in labels:
#         myDetectedFaces.append(label)
# myDetectedFaces = np.array(myDetectedFaces)
# np.savetxt("label_classified_with_double.txt", myDetectedFaces)

# Loading from file if needed
myDetectedFaces = np.loadtxt("label_classified_with_double.txt")
myDetectedFaces = remove_duplicates(myDetectedFaces)
