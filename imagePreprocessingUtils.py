import numpy as np
import cv2
import os
import random
import pickle
from imutils import paths
from scipy.spatial import distance

#PATH or folder name of dataset
PATH = r'C:\Users\Dwarakanath\Indian_Sign_Language_Recognition_main_project\Indian-sign-language-recognition-master\data_v1'
# Train and test factor. 80% is used for training. 20% for testing.
TRAIN_FACTOR = 80
# please reduce TOTAL_IMAGES value to 800 or less if you are facing memory issues.
TOTAL_IMAGES = 1200
# Total number of classes to be classified
N_CLASSES = 35
# clustering factor
CLUSTER_FACTOR = 8

START = (300,75)
END = (600,400) 
##
IMG_SIZE = 128

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
    # computing SIFT descriptors
    kp, des = orb.detectAndCompute(canny,None)
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

# Find the index of the closest central point to the each sift descriptor.   
def find_index(image, center):
    count = 0
    index = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i]) 
        else:
            calculated_distance = distance.euclidean(image, center[i]) 
            if(calculated_distance < count):
                index = i
                count = calculated_distance
    return index
    
# creating labels for imageset
def get_labels():
    class_labels = []
    for (dirpath,dirnames,filenames) in os.walk(PATH):
        dirnames.sort()
        for label in dirnames:
            #print(label)
            if not (label == '.DS_Store'):
                class_labels.append(label)
    
    return class_labels

def concat_tile(im_list_2d, size):
    count = 0
    all_imgs = []
    for row in range(size[1]):
        imgs = []
        for col in range(size[0]):
            imgs.append(im_list_2d[count])
            count += 1
        all_imgs.append(imgs)    
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in all_imgs])

def get_all_gestures():
    gestures = []
    for (dirpath,dirnames,filenames) in os.walk(PATH):
        dirnames.sort()
        for label in dirnames:
            #print(label)
            if not (label == '.DS_Store'):
                for (subdirpath,subdirnames,images) in os.walk(PATH+'/'+label+'/'):
                    random.shuffle(images)
                    #print(label)
                    imagePath = PATH+'/'+label+'/'+images[0]
                    #print(imagePath)
                    img = cv2.imread(imagePath)
                    img = cv2.resize(img, (int(IMG_SIZE * 3/4),int(IMG_SIZE* 3/4)))
                    img = cv2.putText(img, label, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1, cv2.LINE_AA)
                    gestures.append(img)
    
    print('length of gesatures {}'.format(len(gestures)))
    im_tile = concat_tile(gestures, (5, 7))
   # im_tile = concat_tile(gestures, (2, 2))
    
    ''' im_tile = concat_tile([[gestures[0], gestures[1], gestures[2], gestures[3], gestures[4]],
                           [gestures[5], gestures[6], gestures[7], gestures[8], gestures[9]],
                           [gestures[10], gestures[11], gestures[12], gestures[13], gestures[14]],
                           [gestures[15], gestures[16], gestures[17], gestures[18], gestures[19]],
                           [gestures[20], gestures[21], gestures[22], gestures[23], gestures[24]],
                           [gestures[25], gestures[26], gestures[27], gestures[28], gestures[29]],
                           [gestures[30], gestures[31], gestures[32], gestures[33], gestures[34]]])'''
    return im_tile

