import numpy as np
import cv2
from collections import Counter
import os
import pickle
import imagePreprocessingUtils as ipu
import time

CAPTURE_FLAG = False
class_labels = ipu.get_labels()
window_size = 1
# Time when countdown starts
countdown_start_time = 0


multiple_predicts = {}

def recognise(cluster_model, classify_model):
    global CAPTURE_FLAG
    gestures = ipu.get_all_gestures()
    cv2.imwrite("all_gestures.jpg", gestures)
    
    # Open the video capture device
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
    print('Now the camera window will open. \n1) Place your hand gesture in the ROI (rectangle). \n2) Press the " P" key to start / stop recognise . \n3) Press the Esc key to exit.')
    
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break
        
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, ipu.START, ipu.END, (0, 255, 0), 2)
        cv2.imshow("All_gestures", gestures)
        
        # Wait for user input
        pressedKey = cv2.waitKey(1)
        
        # Check for key press events
        if pressedKey == 27:  # ESC key
            break
        # press "P" to start to prediction recognise 
        elif pressedKey == ord('p'):
            CAPTURE_FLAG = not CAPTURE_FLAG
        
        if CAPTURE_FLAG:
            # Region of Interest
            roi = frame[ipu.START[1] + 5:ipu.END[1], ipu.START[0] + 5:ipu.END[0]]
            if roi is not None:
                roi = cv2.resize(roi, (ipu.IMG_SIZE, ipu.IMG_SIZE))
                img = ipu.get_canny_edge(roi)[0]
                cv2.imshow("Edges", img)
                sift_disc = ipu.get_ORB_descriptors(img)
                # sift_disc = = ipu.get_SIFT_descriptors(img)
            
            if sift_disc is not None:
                visual_words = cluster_model.predict(sift_disc)
                # Compute BoVW histogram
                bovw_histogram = np.array(np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR))
                # Predict with confidence using probability estimates
                pred_probabilities = classify_model.predict_proba([bovw_histogram])
                # Get the index of the class with the highest probability
                max_prob_index = np.argmax(pred_probabilities)
                # Get the confidence score associated with the prediction
                confidence_score = pred_probabilities[0][max_prob_index]
                # Get the predicted label
                label = class_labels[max_prob_index]
                # Draw predicted text on the frame
                text = f'Predicted text: {label}'
                # print(f" Predicted = {label} : {confidence_score} %")
                # multiple_predicts[label].append(confidence_score)
                # Define text parameters
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                line_type = cv2.LINE_AA
                text_color = (255, 255, 255)
                background_color = (0, 0, 0)
                
                # Get the size of the text
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                
                # Calculate the position of the filled rectangle
                text_position = (50, 70)
                background_position = (text_position[0], text_position[1] - text_size[1])
                background_end_position = (text_position[0] + text_size[0], text_position[1])
                
                # Draw the filled rectangle
                cv2.rectangle(frame, background_position, background_end_position, background_color, -1)
                
                # Draw the text on top of the filled rectangle
                cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness, line_type)
                
                
                
                                
                                
        cv2.imshow("Video", frame)
    
    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

# Load the clustering and classification models
clustering_model = pickle.load(open('mini_kmeans_model.sav', 'rb'))    
classification_model = pickle.load(open('svm_model.sav', 'rb'))

# Start gesture recognition
recognise(clustering_model, classification_model)
