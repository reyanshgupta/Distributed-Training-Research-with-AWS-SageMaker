import os
import cv2
import numpy as np
import time
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from PIL import Image
from sklearn.metrics import classification_report
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
IMAGE_SIZE = 64  
execution_start_time = time.time()
data = []
labels = []

# Function to prepare the dataset
def prepareDataset(path):
    filePaths = []
    y = []
    count = 0
    name = ''

    for dirname, _, filenames in os.walk(path):
        if count != 0:
            x = dirname.split('/')[-1]
            index = x.rindex('_name_')
            name = x[index+6:].replace(' ','')
        for filename in filenames:
            full_path = os.path.join(dirname, filename)
            filePaths.append(full_path)
            y.append(name)
        
        count += 1
    return filePaths, y

hispanic_path = 'ID_Images_Dataset/Selfies_ID_Images_dataset/11_sets_Hispanics'  
caucasian_path = 'ID_Images_Dataset/Selfies_ID_Images_dataset/18_sets_Caucasians'  

hispanic_filePaths, hispanic_labels = prepareDataset(hispanic_path)
caucasian_filePaths, caucasian_labels = prepareDataset(caucasian_path)


filePaths = hispanic_filePaths + caucasian_filePaths
y_labels = hispanic_labels + caucasian_labels  

# Load and preprocess the images
for i, filePath in enumerate(filePaths):
    if not filePath.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Skipping non-image file: {filePath}")
        continue
    try:
        pil_image = Image.open(filePath).convert('RGB')
        image = np.array(pil_image)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        data.append(image)
        labels.append(y_labels[i])
    except (IOError, OSError) as e:
        print(f"Warning: Could not read image from {filePath} - {e}")
#
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Convert the labels from strings to integers
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Define the number of classes for the CNN model
number_of_classes = labels.shape[1]
print("Classes Count: ",number_of_classes)
np.save('label_classes.npy', lb.classes_)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(number_of_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-4), 
              metrics=['accuracy'])

start_time = time.time()
model.fit(data, labels, epochs=50, batch_size=32)
training_time = time.time() - start_time
print(f"Total training time: {training_time:.2f} seconds")
model.save('face_recognition_model.h5')
model = load_model('face_recognition_model.h5')
predictions = model.predict(data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(labels, axis=1)
lb.classes_ = np.load('label_classes.npy')
report = classification_report(true_classes, predicted_classes, target_names=lb.classes_)
print(report)
end_execution_time = time.time() - execution_start_time
print(f"Total Execution Time: {end_execution_time} seconds")


