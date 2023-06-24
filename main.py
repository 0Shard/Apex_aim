import cv2
from ultralytics import YOLO
import pyautogui
import numpy as np
from pynput import keyboard
from operator import itemgetter
from scipy.spatial import distance

# Load the YOLOv8 model
model = YOLO('apexlegendfinder.pt')

# Specify the class of object to detect
target_class = '0'  # Change this to your target class

# Centering switch
centering = False

# Program running switch
running = True

# IoU and confidence threshold
iou_threshold = 0.65
conf_threshold = 0.5


def on_press(key):
    global centering, running
    try:
        if key.char == 'k':
            centering = not centering
        elif key.char == 'l':
            running = False
    except AttributeError:
        pass


# Setting up keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Run the program as long as 'l' is not pressed
while running:
    # Capture the screenshot
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)

    # Run YOLOv8 inference on the frame
    results = model([frame], conf=conf_threshold, iou=iou_threshold, stream=True)

    if centering:
        for result in results:
            if centering:
                w, h = screenshot.size
                center = np.array([w // 2, h // 2])

                boxes = result.boxes.xyxy  # All bounding boxes

            target_boxes = [(box, cls) for box, cls in zip(boxes, result.boxes.cls) if cls == target_class]
            if not target_boxes:
                continue

            # Find the center of each bounding box, adjusting the bounding box center to pixel values
            box_centers = [(box[:2] + (box[2:4] - box[:2]) / 2) * np.array([w, h]) for box, _ in target_boxes]

            # Calculate Euclidean distances to the center
            distances = [distance.euclidean(center, box_center) for box_center in box_centers]

            # Find the bounding box closest to the center
            min_index = np.argmin(distances)
            closest_box, _ = target_boxes[min_index]

            # Convert from normalized coordinates to pixel values
            box_center = ((closest_box[0] * w + closest_box[2] * w) / 2, (closest_box[1] * h + closest_box[3] * h) / 2)

            # Move the mouse to the center of the bounding box
            pyautogui.moveTo(list(map(int, box_center)))

listener.stop()

# Release the mouse
pyautogui.mouseUp()