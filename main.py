from ultralytics import YOLO
import pyautogui
import numpy as np
from pynput import keyboard
from scipy.spatial import distance
import cv2
import time

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

# Time settings for FPS calculation
prev_frame_time = 0
new_frame_time = 0


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
            pyautogui.moveTo(*map(int, box_center))

            # Draw bounding boxes on the frame
            for box, _ in target_boxes:
                box = (box * np.array([w, h, w, h])).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {int(fps)}"

            # Display FPS on the frame
            cv2.putText(frame, fps_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display the frame with bounding boxes
            cv2.imshow('YOLOv8 Inference', frame)
            cv2.waitKey(1)  # Refresh the display window every 1 millisecond to show the updated image

listener.stop()

# Release the mouse
pyautogui.mouseUp()
