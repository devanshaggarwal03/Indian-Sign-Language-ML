import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
import matplotlib.pyplot as plt
import os

# Define dataset path
dataset_path = "Data"

# Load dataset from directory (automatically labels data based on folder names)
batch_size = 32
img_size = (300, 600)  # Image size (300x600 for both hands)
train_ds = image_dataset_from_directory(dataset_path,shuffle=True,batch_size=batch_size,image_size=img_size)

# Get class names
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Class Names: {class_names}")

# Normalize images (rescale pixel values)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(300, 600, 3)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(num_classes, activation='softmax')  # Output layer
])

# Compile model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train model
epochs = 20
history = model.fit(train_ds, epochs=epochs)

# Save model
model.save("Model/keras_model.h5")

# Save class labels
with open("Model/labels.txt", "w") as f:
    for label in class_names:
        f.write(label + "\n")

# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.title("Model Training Progress")
plt.show()
