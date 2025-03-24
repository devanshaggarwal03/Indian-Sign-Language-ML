import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Detect up to 2 hands
offset = 20
imgSize = 300

# Create folder to store combined images
folder_combined = "Data/G"
os.makedirs(folder_combined, exist_ok=True)

# Variables for automatic saving
last_saved_time = 0
save_interval = 1  # Save every second

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    current_time = time.time()

    # Create a blank white image to store both hands together
    imgCombined = np.ones((imgSize, imgSize * 2, 3), np.uint8) * 255  # 300x600

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Show individual processed images
            cv2.imshow(f'{hand["type"]} Hand Crop', imgCrop)
            cv2.imshow(f'{hand["type"]} Hand Processed', imgWhite)

            # Place the processed hand image in the combined image
            if hand["type"] == "Right":
                imgCombined[:, :imgSize] = imgWhite  # Right hand on left side
            else:
                imgCombined[:, imgSize:] = imgWhite  # Left hand on right side

        # Show combined hand image
        cv2.imshow("Both Hands Combined", imgCombined)

        # Save images automatically every 2 seconds
        if current_time - last_saved_time >= save_interval:
            filename = f'{folder_combined}/Image_{time.time():.0f}.jpg'
            cv2.imwrite(filename, imgCombined)
            print(f"Saved combined image: {filename}")
            last_saved_time = current_time  # Update last saved time

    cv2.imshow('Original Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
