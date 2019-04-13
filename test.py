import numpy as np
from joblib import load
from my_functions import load_from_folder, scan_images_multiprocessed

if __name__ == '__main__':
    print("Loading classifier from \"finalClf.joblib\"")
    clf = load('finalClf.joblib')

    print("Loading test images")
    testImages = load_from_folder("test/")

    print("Finding faces on all images, this may take a while...")
    myDetectedFaces = scan_images_multiprocessed(testImages, clf, 4)

    np.savetxt("detections.txt", myDetectedFaces)
