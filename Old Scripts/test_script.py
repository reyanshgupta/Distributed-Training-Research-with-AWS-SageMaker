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
import argparse

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
            if full_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                filePaths.append(full_path)
                y.append(name)
        count += 1
    return filePaths, y

def main(args):
    hispanic_filePaths, hispanic_labels = prepareDataset(args.hispanic_path)
    caucasian_filePaths, caucasian_labels = prepareDataset(args.caucasian_path)

    filePaths = hispanic_filePaths + caucasian_filePaths
    y_labels = hispanic_labels + caucasian_labels  

    data = []
    labels = []
    # Load and preprocess the images
    for i, filePath in enumerate(filePaths):
        try:
            pil_image = Image.open(filePath).convert('RGB')
            image = np.array(pil_image)
            image = cv2.resize(image, (args.image_size, args.image_size))
            data.append(image)
            labels.append(y_labels[i])
        except (IOError, OSError) as e:
            print(f"Warning: Could not read image from {filePath} - {e}")

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(args.image_size, args.image_size, 3)),
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
        Dense(len(lb.classes_), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    model.fit(data, labels, epochs=args.epochs, batch_size=32)
    model.save(os.path.join(args.model_output_dir, 'face_recognition_model.h5'))

    predictions = model.predict(data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    report = classification_report(true_classes, predicted_classes, target_names=lb.classes_)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hispanic_path', type=str, default=os.environ.get('SM_CHANNEL_TRAINING')+'/11_sets_Hispanics')
    parser.add_argument('--caucasian_path', type=str, default=os.environ.get('SM_CHANNEL_TRAINING')+'/18_sets_Caucasians')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model_output_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()

    main(args)