import os
import cv2
import numpy as np
import time
import argparse
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.optimizers import Adam 
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

def main(args):
    IMAGE_SIZE = 64
    execution_start_time = time.time()
    data = []
    labels = []
    
    def prepareDataset(path):
        filePaths = []
        y = []
        count = 0
        name = ''

        for dirname, _, filenames in os.walk(path):
            if count != 0:
                x = dirname.split('/')[-1]
                index = x.rfind('_name_')  # Changed rindex to rfind
                if index != -1:  # Added a check for index not found
                    name = x[index+6:].replace(' ','')
                else:
                    print(f"Warning: Substring '_name_' not found in directory name: {dirname}")
            for filename in filenames:
                full_path = os.path.join(dirname, filename)
                filePaths.append(full_path)
                y.append(name)

            count += 1
        return filePaths, y
    
    hispanic_path = os.environ.get('SM_CHANNEL_TRAINING', 'distributedml/ID_Images_Dataset/Selfies_ID_Images_dataset/11_sets_Hispanics/')
    caucasian_path =  os.environ.get('SM_CHANNEL_TRAINING', 'distributedml/ID_Images_Dataset/Selfies_ID_Images_dataset/18_sets_Caucasians/')

    hispanic_filePaths, hispanic_labels = prepareDataset(hispanic_path)
    caucasian_filePaths, caucasian_labels = prepareDataset(caucasian_path)

    filePaths = hispanic_filePaths + caucasian_filePaths
    y_labels = hispanic_labels + caucasian_labels  

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
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    number_of_classes = labels.shape[1]
    
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
    history = model.fit(data, labels, epochs=args.epochs)
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    train_accuracy = history.history['accuracy'][-1]
    print(f"Training Accuracy: {train_accuracy:.4f}")

    predictions = model.predict(data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    end_execution_time = time.time() - execution_start_time
    print(f"Total Execution Time: {end_execution_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_output_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args, _ = parser.parse_known_args()
    main(args)
