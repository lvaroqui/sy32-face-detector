import numpy as np
from joblib import load
from skimage import io
from my_functions import load_from_folder, scan_images_multiprocessed, extract_rects, get_label_with_index

if __name__ == '__main__':
    print("Loading classifier from \"finalClf.joblib\"")
    clf = load('finalClf.joblib')

    print("Loading test images")
    testImages = load_from_folder("test/")

    print("Finding faces on all images, this may take a while...")
    myDetectedFaces = scan_images_multiprocessed(testImages, clf, 8, 5, 5, 5)

    j = 0
    for i, image in enumerate(testImages):
        rects = extract_rects(image, get_label_with_index(myDetectedFaces.astype(int), i))
        for rect in rects:
            print("----")
            print(rect)
            io.imsave("test_faces/" + str(j).zfill(4) + ".jpg", rect)
            j += 1

    myDetectedFaces[:, 0] = myDetectedFaces[:, 0] + 1

    np.savetxt("detections.txt", myDetectedFaces)

