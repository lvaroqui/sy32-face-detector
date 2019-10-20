import numpy as np
from joblib import load
from skimage import io
from my_functions import load_from_folder, scan_images_multiprocessed, extract_samples, get_label_with_index

if __name__ == '__main__':
    print("Loading classifier from \"finalClf.joblib\"")
    clf = load('classifiers/svc_kernel_rbf_2/clf.joblib')

    print("Loading test images")
    testImages = load_from_folder("test/")

    print("Finding faces on all images, this may take a while...")
    myDetectedFaces = scan_images_multiprocessed(testImages, clf, processes=8, vstep=8, hstep=8, dnum=10)

    # Adjusting indexes to correspond to image number
    myDetectedFaces[:, 0] = myDetectedFaces[:, 0] + 1

    np.savetxt("detections.txt", myDetectedFaces)

