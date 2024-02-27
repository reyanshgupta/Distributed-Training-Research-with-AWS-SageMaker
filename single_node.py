import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import time
start_time = time.time()

dataset_path = 'FaceDisguiseDatabase/FaceAll'

data = []
labels = []

image_count = 0
person_count = 0
current_label = None

for image_file in os.listdir(dataset_path):
    if image_count >= 6:  
        image_count = 0
        person_count += 1
        continue

    image_path = os.path.join(dataset_path, image_file)

    if os.path.isdir(image_path): 
        continue

    if os.path.isfile(image_path):
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            data.append(image)
            if image_count == 0: 
                current_label = f"Person_{person_count}"
            labels.append(current_label)

            image_count += 1
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

data = np.array(data, dtype="uint8")
labels = np.array(labels)
data = data / 255.0
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = to_categorical(encoded_labels)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
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

epochs = 5
checkpoint_path = "best_model.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit(data, one_hot_labels, epochs=epochs, validation_split=0.1, callbacks=[checkpoint])

model.save("final_model.h5")
if os.path.exists(checkpoint_path):
    best_model = models.load_model(checkpoint_path)
    best_loss, best_acc = best_model.evaluate(data, one_hot_labels, verbose=2)
    print(f'Best model accuracy: {best_acc}')
    predictions = best_model.predict(data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(one_hot_labels, axis=1)
    unique_labels = np.unique(np.concatenate((predicted_classes, true_classes)))
    target_names = label_encoder.inverse_transform(unique_labels)
    
    report = classification_report(true_classes, predicted_classes, labels=unique_labels, target_names=target_names)
    print(report)
else:
    print("Best model checkpoint not found.")

end_time = time.time()
total_time = end_time - start_time
print("Total time taken:", total_time)