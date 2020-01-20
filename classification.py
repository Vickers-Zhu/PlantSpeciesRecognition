import numpy as np
import pandas as pd
import os
import cv2
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import mahotas as mt
from matplotlib import pyplot as plt

common_names = ['garryana', 'glabrum', 'macrophyllum',
                'kelloggii', 'circinatum', 'negundo']

ds_path = "../isolated"
X = pd.read_csv("Leaves.csv")

i = 0
target_list = []
for dir in os.listdir(ds_path):
    image_dirs = os.path.join(os.path.join(ds_path, dir))
    if os.path.isdir(image_dirs):
        for file in os.listdir(image_dirs):
            target_list.append(i)
        i = i + 1
y = np.array(target_list)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=142)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
             ]

svm_clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
svm_clf.fit(X_train, y_train)

means = svm_clf.cv_results_['mean_test_score']
stds = svm_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svm_clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_pred_svm = svm_clf.predict(X_test)

pca = PCA()
pca.fit(X)
var = pca.explained_variance_ratio_
var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


def bg_sub(filename):
    test_img_path = filename
    main_img = cv2.imread(test_img_path)
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(img, (1600, 1200))
    size_y, size_x, _ = img.shape
    gs = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (55, 55), 0)
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contains = []
    y_ri, x_ri, _ = resized_image.shape
    for cc in contours:
        yn = cv2.pointPolygonTest(cc, (x_ri // 2, y_ri // 2), False)
        contains.append(yn)

    val = [contains.index(temp) for temp in contains if temp > 0]
    index = val[0]

    black_img = np.empty([1200, 1600, 3], dtype=np.uint8)
    black_img.fill(0)

    cnt = contours[index]
    mask = cv2.drawContours(black_img, [cnt], 0, (255, 255, 255), -1)

    maskedImg = cv2.bitwise_and(resized_image, mask)
    white_pix = [255, 255, 255]
    black_pix = [0, 0, 0]

    final_img = maskedImg
    h, w, channels = final_img.shape
    for x in range(0, w):
        for y in range(0, h):
            channels_xy = final_img[y, x]
            if all(channels_xy == black_pix):
                final_img[y, x] = white_pix

    return final_img


def feature_extract(img):
    names = ['area', 'perimeter', 'pysiological_length', 'pysiological_width', 'aspect_ratio', 'rectangularity',
             'circularity', \
             'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b', \
             'contrast', 'correlation', 'inverse_difference_moments', 'entropy'
             ]
    df = pd.DataFrame([], columns=names)

    # Preprocessing
    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25, 25), 0)
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    # Shape features
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    rectangularity = w * h / area
    circularity = ((perimeter) ** 2) / area

    # Color features
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    # Texture features
    textures = mt.features.haralick(gs)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]

    vector = [area, perimeter, w, h, aspect_ratio, rectangularity, circularity, \
              red_mean, green_mean, blue_mean, red_std, green_std, blue_std, \
              contrast, correlation, inverse_diff_moments, entropy
              ]

    df_temp = pd.DataFrame([vector], columns=names)
    df = df.append(df_temp)
    return df


# bg_rem_img = bg_sub('l16.jpg')
features_of_img = feature_extract(cv2.imread('l16.jpg'))
scaled_features = sc_X.transform(features_of_img)
# y_pred_mobile = svm_clf.predict(features_of_img)
y_pred_mobile = svm_clf.predict(scaled_features)
print(common_names[y_pred_mobile[0]])

