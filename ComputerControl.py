import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyautogui  # for controlling mouse/keyboard
import time
import os
from scipy import stats
import pygetwindow as gw
import win32gui
import win32con

# Make OpenCV window "OpenCV Feed" always on top
try:
    hwnd = gw.getWindowsWithTitle('OpenCV Feed')[0]._hWnd
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
except IndexError:
    pass  # Window not found (yet)

mp_holistic = mp.solutions.holistic
mp_drawings = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):  
    #Draw left hand connections
    mp_drawings.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawings.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                               mp_drawings.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    #Draw right hand connections 
    mp_drawings.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawings.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),  
                               mp_drawings.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))  
    

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh,rh])

# Load your trained model
model = load_model('CC_action.h5')  # Replace with your .h5 file path

colors = [
    (245, 117, 16),   # Orange
    (117, 245, 16),   # Green
    (16, 117, 245),   # Blue
    (255, 0, 255),    # Magenta
    (0, 255, 255)     # Cyan
]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

import pyautogui
import webbrowser
import time
import random
import math

def perform_action(prediction):
    if prediction == 'click':
        pyautogui.click()
        print("Clicked")

    elif prediction == 'scroll':
        pyautogui.scroll(-500)  # scroll down; use +500 to scroll up
        print("Scrolled")

    elif prediction == 'swipe':
        # Swipe cursor around the screen in a smooth circular motion
        x, y = pyautogui.position()
        for angle in range(0, 360, 30):
            radius = 100
            new_x = x + int(radius * math.cos(math.radians(angle)))
            new_y = y + int(radius * math.sin(math.radians(angle)))
            pyautogui.moveTo(new_x, new_y, duration=0.05)
        print("Swiped")

    elif prediction == 'pause':
        pyautogui.press('space')  # commonly pauses/resumes videos
        print("Paused")

    elif prediction == 'tabchange':
        pyautogui.hotkey('alt', 'tab')  # switch to next tab
        print("Tab Changed")

    else:
        print("Unknown Action")


last_prediction = ''
last_time = 0
cooldown = 2  # seconds

def should_trigger(pred, conf, threshold=0.8):
    global last_prediction, last_time
    if conf > threshold and (pred != last_prediction or time.time() - last_time > cooldown):
        last_prediction = pred
        last_time = time.time()
        return True
    return False


sequence = []
sentence = []
threshold = 0.7

actions = np.array(['click','scroll','swipe','pause','tabchange'])
res = np.zeros(len(actions))  # Initialize res to zeros



cap = cv2.VideoCapture(0)
#Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Make OpenCV window "OpenCV Feed" always on top
        try:
            hwnd = gw.getWindowsWithTitle('OpenCV Feed')[0]._hWnd
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        except IndexError:
            pass  # Window not found (yet)

        # Read feed
        ret, frame = cap.read()

        # MAke Decision
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_landmarks(image, results)

        if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
            cv2.putText(image, "No hands detected", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
            continue

        # Prediction Logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Trigger Action
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence,axis=0))[0]
            print(actions[np.argmax(res)])

            conf = np.max(res)
            pred_class = actions[np.argmax(res)]

            if should_trigger(pred_class, conf):
                perform_action(pred_class)

        # viz logic
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                        sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]

        # viz probabilities
        image = prob_viz(res,actions,image,colors)
        
        cv2.rectangle(image,(0,0),(640,40),(245,117,16),-1)
        cv2.putText(image, ' '.join(sentence),(3,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)

        # Show to Screen
        cv2.imshow('OpenCV Feed', image)

        # Break the code
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()