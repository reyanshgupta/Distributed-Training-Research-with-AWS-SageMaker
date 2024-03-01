import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

start_time = time.time()
dataset_path = 'FaceDisguiseDatabase/FaceAll_cropped'
data = []
labels = []
image_count = 0
person_count = 0
current_label = None

for image_file in os.listdir(dataset_path):
    if image_count == 6:  
        image_count = 0
        person_count += 1
        continue

    image_path = os.path.join(dataset_path, image_file)
    if os.path.isdir(image_path): 
        continue
    if os.path.isfile(image_path):
        try:
            image = cv2.imread(image_path, 0)  # Load as grayscale
            image = cv2.resize(image, (64, 64))
            data.append(image)
            current_label = f"Person_{person_count}"
            labels.append(current_label)
            image_count += 1
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image_count += 1

data = np.array(data, dtype="uint8")
labels = np.array(labels)
# No need to normalize grayscale images
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = to_categorical(encoded_labels)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data, one_hot_labels, test_size=0.2, random_state=42)
# Reshape input data to match input shape
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(encoded_labels)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 50
batch_size = 32
learning_rate = 0.001
checkpoint_path = "best_model.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint]
)

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(X_val, y_val, verbose=2)
print(f'Validation Accuracy: {acc*100:.2f}%')

end_time = time.time()
total_time = end_time - start_time
print("Total time taken: {:.2f} seconds".format(total_time))
