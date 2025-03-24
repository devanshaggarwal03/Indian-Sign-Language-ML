import cv2
import numpy as np
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import math

# Load the trained Keras model
model = load_model("Model/keras_model.h5")

# Print model summary to debug input shape
model.summary()

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

# Define image size for model input
imgSize = 300
offset = 20

# Define class labels (update as per your trained model)
labels = ["A", "B", "C", "D", "E", "F", "G"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            
            try:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]), max(0, x - offset):min(x + w + offset, img.shape[1])]
                
                if imgCrop.size == 0:
                    continue  # Skip processing if the cropped image is empty

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

                # Ensure the image is in RGB format
                inputSize = model.input_shape[1:3]  # Get model's expected input size
                imgInput = cv2.resize(imgWhite, inputSize)  # Resize to match model input
                imgInput = imgInput / 255.0  # Normalize pixel values
                imgInput = np.expand_dims(imgInput, axis=0)  # Add batch dimension
                
                # Flatten input if required by model
                if len(model.input_shape) == 2:
                    imgInput = imgInput.reshape(1, -1)  # Flatten input

                # Make a prediction
                prediction = model.predict(imgInput)
                print(f"Raw Model Output: {prediction}")  # Debugging output
                predicted_index = np.argmax(prediction)
                predicted_label = labels[predicted_index]
                print(f"Predicted Label: {predicted_label}")  # Debugging output

                # Display prediction
                cv2.putText(img, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Processed Image {hand['type']}", imgWhite)
            
            except Exception as e:
                print(f"Error processing hand: {e}")  # Print error message
    
    cv2.imshow("Original Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
