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
from my_functions import scan_images_multiprocessed
from my_functions import stats
from my_functions import get_label_with_index
from joblib import dump

from my_functions import precision_rappel
from matplotlib import pyplot as plt

from os import mkdir

from tqdm import tqdm

if __name__ == '__main__':

    # CONFIGURATION

    from sklearn.svm import SVC
    # from sklearn.svm import LinearSVC

    config = "svc_kernel_3"
    comment = "new scale system"

    clf = SVC(kernel="rbf", gamma="scale")
    clf2 = SVC(kernel="rbf", gamma="scale")
    # clf = LinearSVC(C=3)
    # clf2 = LinearSVC(C=3)

    clf.max_iter = 3000
    clf2.max_iter = 3000

    verticalStep = 10
    horizontalStep = 10
    divideScale = 10

    processes = 8

    # PROGRAM

    # IMPORTANT NOTE : I renamed the train image and label n°1000 to n°0
    # to have picture index matching their number

    I_INDEX = 1
    J_INDEX = 2
    HEIGHT_INDEX = 3
    WIDTH_INDEX = 4

    print("----Preparing data for classifier----")
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

    # print("Generating negative patches...")
    # random_per_image = 10
    # myNegatives = np.zeros((len(myImages) * random_per_image, 60, 40, 3))
    # for i in tqdm(range(len(myImages))):
    #     myNegatives[i * random_per_image:i * random_per_image + 10] \
    #         = generate_random_negatives(random_per_image, myImages[i], get_label_with_index(myLabelsResized, i))
    #
    # print("Saving negative patches to disk...")
    # for i, negative in enumerate(tqdm(util.img_as_ubyte(myNegatives))):
    #     io.imsave("negatives/" + str(i).zfill(5) + ".jpg", negative)

    print("Loading negatives from folder...")
    myNegatives = load_from_folder("negatives/")

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

    print("\n----Intermediate classifier----")
    print("Learning...")

    clf.fit(myShuffledHogs, myShuffledTargets)

    from sklearn.model_selection import cross_validate

    print("Cross validating classifier...")
    cv_results = cross_validate(clf, myShuffledHogs, myShuffledTargets, cv=3, return_train_score=False)
    print("Result of cross_validation :")
    print(cv_results["test_score"])

    print("Finding faces on all images, this may take a while...")
    myDetections = scan_images_multiprocessed(myImages,
                                              clf,
                                              processes,
                                              verticalStep,
                                              horizontalStep,
                                              divideScale).astype(int)

    print("\nStatistics for intermediate classifier (precision, rappel and f-score :")
    print(stats(myDetections, myLabelsResized))

    print("\n----Extracting intermediate classifier false positives----")

    print("Extracting false positives labels from detections...")
    myFalsePositivesLabels = []
    for i in tqdm(range(myDetections[-1, 0] + 1)):
        detections = get_label_with_index(myDetections, i)
        faces = get_label_with_index(myLabelsResized, i)
        myFalsePositivesLabels += get_false_positives(detections, faces)

    myFalsePositivesLabels = np.array(myFalsePositivesLabels)

    print("Extracting false positives from images...")
    myFalsePositives = np.zeros((len(myFalsePositivesLabels), 60, 40, 3))
    i = 0
    for image_index, image in enumerate(tqdm(myImages)):
        false_positives_for_image = extract_rects(image, myFalsePositivesLabels[
            np.where(myFalsePositivesLabels[:, 0] == image_index)])
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

    print("\n----Final Classifier----")
    print("Learning with False Positives from intermediate Classifer...")
    clf2.fit(myShuffledHogsWithFalsePositives, myShuffledTargetsWithFalsePositives)

    print("Cross validating classifier...")
    cv_results = cross_validate(clf2, myShuffledHogsWithFalsePositives, myShuffledTargetsWithFalsePositives, cv=3,
                                return_train_score=False)
    print("Result of cross_validation :")
    print(cv_results["test_score"])

    print("Finding faces on all images with final classifier, this may take a while...")
    myNewDetections = scan_images_multiprocessed(myImages, clf2, processes, verticalStep, horizontalStep, divideScale)

    print("\nStatistics for final classifier (precision, rappel and f-score) :")
    myNewDetectionsStats = stats(myNewDetections, myLabelsResized)
    print(myNewDetectionsStats)

    print("\nSaving data...")
    folder_path = "results/" + config + "/"
    mkdir(folder_path)
    np.savetxt(folder_path + "detections.txt", myNewDetections)
    dump(clf2, folder_path + 'clf.joblib')

    f = open(folder_path + "summary.txt", "x")
    f.write(config + "\n")
    f.write("Comment: {}".format(comment) + "\n")
    f.write("Vertical step: {}".format(verticalStep) + "\n")
    f.write("Horizontal step: {}".format(horizontalStep) + "\n")
    f.write("Divider Scale: {}".format(divideScale) + "\n\n")
    f.write("Precision: {}".format(myNewDetectionsStats[0]) + "\n")
    f.write("Rappel: {}".format(myNewDetectionsStats[1]) + "\n")
    f.write("F-Score: {}".format(myNewDetectionsStats[2]) + "\n")
    f.close()

    precision_rappel_data = precision_rappel(myNewDetections, myLabelsResized)
    plt.plot(precision_rappel_data[:, 0], precision_rappel_data[:, 1], linewidth=5)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig(folder_path + "precision_rappel.jpg")