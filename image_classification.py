import numpy as np
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

IMG_WIDTH = 25
IMG_HEIGHT = 25


def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            image = image.flatten()
            img_data_array.append(image)
            class_name.append(dir1)

    img_data_array = np.asarray(img_data_array)
    class_name = np.asarray(class_name)
    return img_data_array, class_name

def load_image(image_path):
    arr = []
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    image = image.flatten()
    arr.append(image)
    arr = np.asarray(arr)
    return arr

# def data_from_2D_to_1D(data):
#     dataset_size = data.shape[0]
#     data = data.reshape(dataset_size, -1)  # we convert the image from 2d to 1d
#     return data


def under_sample(X, y):
    # define undersample strategy
    undersample = RandomUnderSampler(random_state=42)
    # fit and apply the transform
    X_over, y_over = undersample.fit_resample(X, y)
    return X_over, y_over


def print_report(trainY, testY, testX, encoder, model):
    print("train class distribution")
    unique, counts = np.unique(trainY, return_counts=True)
    print(dict(zip(unique, counts)))
    print("report")
    print(classification_report(testY, model.predict(testX), target_names=encoder.classes_))


def test_knn(data, labels, max_k_size, encoder):
    print("start knn algorithm")

   # data = data_from_2D_to_1D(data)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, train_size=0.70, test_size=0.30, random_state=42)

    error = []
    for k in range(1, max_k_size, 2):

        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

        model.fit(trainX, trainY)

        pred_i = model.predict(testX)
        error.append(np.mean(pred_i != testY))

        if k == 15:  # print a report for k = 15
            print_report(trainY, testY, testX, encoder, model)

    x = np.arange(1, max_k_size, 2)
    plt.title("knn")
    plt.xlabel("neighbors")
    plt.ylabel("error")
    plt.plot(x, error, color="red")
    plt.show()

    errors_e = []  # mean train errors
    errors_i = []  # mean test errors

    for i in range(60, 90, 5):
        (trainX, testX, trainY, testY) = train_test_split(data, labels, train_size=i / 100, test_size=1 - (i / 100),
                                                          random_state=42)

        # trainX, trainY = under_sample(trainX, trainY) # under sample if needed

        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        model.fit(trainX, trainY)

        pred_e = model.predict(trainX)
        error_e = np.mean(pred_e != trainY)

        pred_i = model.predict(testX)
        error_i = np.mean(pred_i != testY)

        errors_e.append(error_e)
        errors_i.append(error_i)

        if i == 80:
            print_report(trainY, testY, testX, encoder, model)

    x = np.arange(60, 90, 5)
    plt.title("knn")
    plt.xlabel("train %")
    plt.ylabel("error")
    plt.plot(x, errors_e, color="red")
    plt.plot(x, errors_i, color="blue")
    plt.show()

    # test image of "5"
    image = load_image(r'test_img.png')
    pred = model.predict(image)
    print(pred)


def test_dt(data, labels, encoder):
    print("start decision trees algorithm")

    #data = data_from_2D_to_1D(data)

    errors_e = []  # mean train errors
    errors_i = []  # mean test errors

    for i in range(60, 90, 5):
        (trainX, testX, trainY, testY) = train_test_split(data, labels, train_size=i / 100, test_size=1 - (i / 100),
                                                          random_state=42)

        # trainX, trainY = under_sample(trainX, trainY) # under sample if needed

        model = DecisionTreeClassifier(criterion="gini")
        model.fit(trainX, trainY)

        pred_e = model.predict(trainX)
        error_e = np.mean(pred_e != trainY)

        pred_i = model.predict(testX)
        error_i = np.mean(pred_i != testY)

        errors_e.append(error_e)
        errors_i.append(error_i)

        if i == 80:
            print_report(trainY, testY, testX, encoder, model)

    x = np.arange(60, 90, 5)
    plt.title("decision trees")
    plt.xlabel("train %")
    plt.ylabel("error")
    plt.plot(x, errors_e, color="red")
    plt.plot(x, errors_i, color="blue")
    plt.show()

    # check optimal max depth

    errors_i = []  # mean test  errors

    for i in range(25, 51, 1):
        (trainX, testX, trainY, testY) = train_test_split(data, labels, train_size=0.8, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier(criterion="gini", max_depth=i)
        model.fit(trainX, trainY)

        pred_i = model.predict(testX)
        error_i = np.mean(pred_i != testY)
        errors_i.append(error_i)

    x = np.arange(25, 51, 1)
    plt.title("decision trees")
    plt.xlabel("max_depth")
    plt.ylabel("error")
    plt.plot(x, errors_i, color="blue")
    plt.show()

    # test image of "5"
    image = load_image(r'test_img.png')
    pred = model.predict(image)
    print(pred)




def test_svm(data, labels, encoder):
    print("start SVM algorithm")

    #data = data_from_2D_to_1D(data)

    errors_e = []  # mean train errors
    errors_i = []  # mean test errors

    for i in range(60, 90, 5):
        (trainX, testX, trainY, testY) = train_test_split(data, labels, train_size=i / 100, test_size=1 - (i / 100),
                                                          random_state=42)
        model = SVC(decision_function_shape='ovr')
        model.fit(trainX, trainY)

        pred_e = model.predict(trainX)
        error_e = np.mean(pred_e != trainY)

        pred_i = model.predict(testX)
        error_i = np.mean(pred_i != testY)

        errors_e.append(error_e)
        errors_i.append(error_i)

        if i == 80:
            print_report(trainY, testY, testX, encoder, model)

    x = np.arange(60, 90, 5)
    plt.title("SVC")
    plt.xlabel("train %")
    plt.ylabel("error")
    plt.plot(x, errors_e, color="red")
    plt.plot(x, errors_i, color="blue")
    plt.show()

    # test image of "5"
    image = load_image(r'test_img.png')
    pred = model.predict(image)
    print(pred)


def test_mlp(data, labels, encoder):
    print("start MLP algorithm")

    #data = data_from_2D_to_1D(data)
    errors_e = []  # mean train errors
    errors_i = []  # mean test errors

    for i in range(60, 90, 5):
        (trainX, testX, trainY, testY) = train_test_split(data, labels, train_size=i / 100, test_size=1 - (i / 100),
                                                          random_state=42)
        model = MLPClassifier(activation='relu', max_iter=500, solver='adam', random_state=42)
        model.fit(trainX, trainY)

        pred_e = model.predict(trainX)
        error_e = np.mean(pred_e != trainY)

        pred_i = model.predict(testX)
        error_i = np.mean(pred_i != testY)

        errors_e.append(error_e)
        errors_i.append(error_i)

        if i == 9:
            print_report(trainY, testY, testX, encoder, model)

    x = np.arange(60, 90, 5)
    plt.title("MLP")
    plt.xlabel("train %")
    plt.ylabel("error")
    plt.plot(x, errors_e, color="red")
    plt.plot(x, errors_i, color="blue")
    plt.show()


def main():
    img_folder = r'dataset'
    data, labels = create_dataset(img_folder)

    le = LabelEncoder()
    labels = le.fit_transform(labels)  # encodes the labels into numerical values

    dataset_size = data.shape[0]
    data = data.reshape(dataset_size, -1)  # we convert the image from 2d to 1d

    # show how many images of each class we have
    print("class distribution")
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))
    print("-----------------")

    test_knn(data, labels, 29, le)
    test_dt(data, labels, le)
    test_svm(data, labels, le)
    test_mlp(data, labels, le)




if __name__ == '__main__':
    main()
