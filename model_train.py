import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

import cv2
print(cv2.__version__)

# Set up data paths (corrected paths)
train_dir = r'C:\Users\FINRISE\Desktop\cat_dog_classifier\training_set\training_set'
test_dir = r'C:\Users\FINRISE\Desktop\cat_dog_classifier\test_set\test_set'

image_size = (180, 180)
batch_size = 32

# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

# Train data flow
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Test data flow
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and save the model
model.fit(train_data, epochs=10, validation_data=test_data)
model.save('cat_dog_classifier_model.h5')
print("✅ Model saved as cat_dog_classifier_model.h5")
