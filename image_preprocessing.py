#imports
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skmetrics
import random
import pickle
import imagePreprocessingUtils as ipu

#import glob

train_labels = []
test_labels = []

# function for get canny image
def get_canny_edge(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

    # Define the lower and upper boundaries for skin color in HSV
    lower_boundary = np.array([0, 40, 30], dtype=np.uint8)
    upper_boundary = np.array([43, 255, 254], dtype=np.uint8)

    # Create a binary mask for skin detection
    skin_mask = cv2.inRange(hsv_image, lower_boundary, upper_boundary)
    
    # Blur the grayscale image using medianBlur
    skin_mask = cv2.addWeighted(skin_mask, 0.5, skin_mask, 0.5, 0.0)
    skin_mask = cv2.medianBlur(skin_mask, 5)
    
    # Apply bitwise AND operation to the grayscale image and the skin mask
    skin = cv2.bitwise_and(gray_image, gray_image, mask=skin_mask)
    
    # Apply Canny edge detection to the skin mask
    canny_edges = cv2.Canny(skin, 60, 60)
    
    return canny_edges, skin

# function for ORB Algorithm
def get_ORB_descriptors(canny):
    
   # Initialize ORB
    orb = cv2.ORB_create()
    # Resize the Canny edge image to a standard size
    canny = cv2.resize(canny, (256, 256))
    # computing ORB descriptors
    kp, des = orb.detectAndCompute(canny, None)
    return des

# function for SIFT Algorithm
def get_SIFT_descriptors(canny):
    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Resize the Canny edge image to a standard size
    canny = cv2.resize(canny, (256, 256))
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(canny, None)
    return descriptors

def preprocess_all_images():
    images_labels = []
    train_disc_by_class = {}
    test_disc_by_class = {}
    all_train_dis = []
    train_img_disc = []
    test_img_disc = []
    label_value = 0
    for (dirpath,dirnames,filenames) in os.walk(ipu.PATH):
        dirnames.sort()
        for label in dirnames:
            #print(label)
            if not (label == '.DS_Store'):
                for (subdirpath,subdirnames,images) in os.walk(ipu.PATH+'/'+label+'/'):
                    #print(len(images))
                    count = 0
                    train_features = []
                    test_features = []
                    for image in images: 
                        #print(label)
                        imagePath = ipu.PATH+'/'+label+'/'+image
                        #print(imagePath)
                        img = cv2.imread(imagePath)
                        if img is not None:
                            img = get_canny_edge(img)[0]
                            sift_disc = get_ORB_descriptors(img)
                            # sift_disc = get_SIFT_descriptors(img)
                            print(sift_disc.shape)
                            if(count < (ipu.TOTAL_IMAGES * ipu.TRAIN_FACTOR * 0.01)):
                                print('Train:--------- Label is {} and Count is {}'.format(label, count)  )
                                #train_features.append(sift_disc)
                                train_img_disc.append(sift_disc)
                                all_train_dis.extend(sift_disc)
                                train_labels.append(label_value)
                            elif((count>=(ipu.TOTAL_IMAGES * ipu.TRAIN_FACTOR  * 0.01)) and count <ipu.TOTAL_IMAGES):
                                print('Test:--------- Label is {} and Count is {}'.format(label, count)  )
                                #test_features.append(sift_disc)
                                test_img_disc.append(sift_disc)
                                test_labels.append(label_value)
                            count += 1
                        #images_labels.append((label,sift_disc))
                #train_disc_by_class[label] = train_features
                #test_disc_by_class[label] = test_features
                label_value +=1
                    
    print('length of train features are %i' % len(train_img_disc))
    print('length of test features are %i' % len(test_img_disc))
    print('length of all train discriptors is {}'.format(len(all_train_dis)))
    #print('length of all train discriptors by class  is {}'.format(len(train_disc_by_class)))
    #print('length of all test disc is {}'.format(len(test_disc_by_class))) 
    return all_train_dis, train_img_disc, train_disc_by_class, test_disc_by_class, test_img_disc


### K-means is not used as data is large and requires a better computer with good specifications
def kmeans(k, descriptor_list):
    print('K-Means started.')
    print ('%i descriptors before clustering' % descriptor_list.shape[0])
    kmeanss = KMeans(k)
    kmeanss.fit(descriptor_list)
    visual_words = kmeanss.cluster_centers_ 
    return visual_words, kmeans

def mini_kmeans(k, descriptor_list):
    print('Mini batch K-Means started.')
    print ('%i descriptors before clustering' % descriptor_list.shape[0])
    kmeans_model = MiniBatchKMeans(k)
    kmeans_model.fit(descriptor_list)
    print('Mini batch K means trained to get visual words.')
    filename = 'mini_kmeans_model.sav'
    pickle.dump(kmeans_model, open(filename, 'wb'))
    return kmeans_model

def get_histograms(discriptors_by_class,visual_words, cluster_model):
    histograms_by_class = {}
    total_histograms = []
    for label,images_discriptors in discriptors_by_class.items():
        print('Label: %s' % label)
        histograms = []
        #    loop for all images 
        for each_image_discriptors in images_discriptors:
            
            ## manual method to calculate words occurence as histograms
            '''histogram = np.zeros(len(visual_words))
            # loop for all discriptors in a image discriptorss 
            for each_discriptor in each_image_discriptors:
                #list_words = visual_words.tolist()
                a = np.array([visual_words])
                index = find_index(each_discriptor, visual_words)
                #print(index)
                #del list_words
                histogram[index] += 1
            print(histogram)'''
            
            ## using cluster model
            raw_words = cluster_model.predict(each_image_discriptors)
            hist =  np.bincount(raw_words, minlength=len(visual_words))
            print(hist)
            histograms.append(hist)
        histograms_by_class[label] = histograms
        total_histograms.append(histograms)
    print('Histograms succesfully created for %i classes.' % len(histograms_by_class))
    return histograms_by_class, total_histograms
    
def dataSplit(dataDictionary):
    X = []
    Y = []
    for key,values in dataDictionary.items():
        for value in values:
            X.append(value)
            Y.append(key)
    return X,Y

# for svm algorithm function 
def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear',probability=True) 
    print("Support Vector Machine started.")
    svc.fit(X_train,y_train)
    filename = 'svm_model.sav'
    pickle.dump(svc, open(filename, 'wb'))
    y_pred=svc.predict(X_test)
    np.savetxt('submission_svm.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,PredictedLabel,TrueLabel', comments = '', fmt='%d')
    calculate_metrics("SVM",y_test,y_pred)

# for random forest algorithm
def predict_random_forest(X_train, X_test, y_train, y_test):
    # Initialize Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust hyperparameters here
    
    print("Random Forest started.")
    
    # Train the Random Forest classifier
    rf_clf.fit(X_train, y_train)
    
    # Save the trained model to a file
    filename = 'random_forest_model.sav'
    pickle.dump(rf_clf, open(filename, 'wb'))
    
    # Predict labels for test data
    y_pred = rf_clf.predict(X_test)
    
    # Save predictions to a CSV file
    np.savetxt('submission_random_forest.csv', np.c_[range(1, len(y_test)+1), y_pred, y_test], delimiter=',', 
               header='ImageId,PredictedLabel,TrueLabel', comments='', fmt='%d')
    
    # Calculate and print evaluation metrics
    calculate_metrics("Random Forest", y_test, y_pred)

def calculate_metrics(method,label_test,label_pred):
    print("Accuracy score for ",method,skmetrics.accuracy_score(label_test,label_pred))
    print("Precision_score for ",method,skmetrics.precision_score(label_test,label_pred,average='micro'))
    print("f1 score for ",method,skmetrics.f1_score(label_test,label_pred,average='micro'))
    print("Recall score for ",method,skmetrics.recall_score(label_test,label_pred,average='micro'))
    
### STEP:1 ORB discriptors for all train and test images with class seperation

all_train_dis,train_img_disc, train_disc_by_class, test_disc_by_class, test_img_disc  = preprocess_all_images()

##  deleting these variables as they are not used with mini batch k means
del train_disc_by_class, test_disc_by_class 

### STEP:2 MINI K-MEANS 

mini_kmeans_model = mini_kmeans(ipu.N_CLASSES * ipu.CLUSTER_FACTOR, np.array(all_train_dis))

del all_train_dis

### Collecting VISUAL WORDS for all images (train , test)

print('Collecting visual words for train .....')
train_images_visual_words = [mini_kmeans_model.predict(visual_words) for visual_words in train_img_disc]
print('Visual words for train data collected. length is %i' % len(train_images_visual_words))

print('Collecting visual words for test .....')
test_images_visual_words = [mini_kmeans_model.predict(visual_words) for visual_words in test_img_disc]
print('Visual words for test data collected. length is %i' % len(test_images_visual_words))


### STEP:3 HISTOGRAMS (findiing the occurence of each visual word of images in total words)
## Can be calculated using get_histograms function also manually

print('Calculating Histograms for train...')
bovw_train_histograms = np.array([np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR) for visual_words in train_images_visual_words])
print('Train histograms are collected. Length : %i ' % len(bovw_train_histograms))

print('Calculating Histograms for test...')
bovw_test_histograms = np.array([np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR) for visual_words in test_images_visual_words])
print('Test histograms are collected. Length : %i ' % len(bovw_test_histograms))

print('Each histogram length is : %i' % len(bovw_train_histograms[0]))
#----------------------
print('============================================')

# preperaing for training svm
X_train = bovw_train_histograms
X_test = bovw_test_histograms
Y_train = train_labels
Y_test = test_labels

#print(Y_train)
### shuffling 

buffer  = list(zip(X_train, Y_train))
random.shuffle(buffer)
random.shuffle(buffer)
random.shuffle(buffer)
X_train, Y_train = zip(*buffer)
#print(Y_train)

buffer  = list(zip(X_test, Y_test))
random.shuffle(buffer)
random.shuffle(buffer)
X_test, Y_test = zip(*buffer)

print('Length of X-train:  %i ' % len(X_train))
print('Length of Y-train:  %i ' % len(Y_train))
print('Length of X-test:  %i ' % len(X_test))
print('Length of Y-test:  %i ' % len(Y_test))

predict_svm(X_train, X_test,Y_train, Y_test)
# predict_random_forest(X_train, X_test,Y_train, Y_test)