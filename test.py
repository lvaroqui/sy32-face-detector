import numpy as np
from skimage import feature
from skimage import transform
from skimage import util
from skimage.color import rgb2gray

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from joblib import load
from tqdm import tqdm


from my_functions import load_from_folder
from my_functions import resize_label
from my_functions import extract_rects
from my_functions import intersection_ratio
from my_functions import generate_random_negatives
from my_functions import find_faces
from my_functions import _remove_duplicates

print("Loading classifier from \"clf2.joblib\"")
clf = load('clf2.joblib')

print("Loading test images")
testImages = load_from_folder("test/")

print("Finding faces on all images, this may take a while...")
myDetectedFaces = []
for i, image in enumerate(tqdm(testImages)):
    labels = find_faces(util.img_as_float(rgb2gray(image)), clf, i+1)
    for label in labels:
        myDetectedFaces.append(label)

myDetectedFaces = np.array(myDetectedFaces)
np.savetxt("detections.txt", myDetectedFaces)
