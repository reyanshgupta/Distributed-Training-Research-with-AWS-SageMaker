import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse
import time

def build_and_compile_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    start_training_time = time.time()

    # Load dataset from the input directory
    input_data_path = os.environ.get('SM_CHANNEL_TRAINING', 'distributedmachinelearning/FaceAll/')
    
    data = []
    labels = []

    image_count = 0
    person_count = 0
    current_label = None

    for image_file in os.listdir(input_data_path):
        if image_count >= 6:
            image_count = 0
            person_count += 1
            continue

        image_path = os.path.join(input_data_path, image_file)
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
                image_count += 1

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    one_hot_labels = to_categorical(encoded_labels)

    model = build_and_compile_model((64, 64, 3), num_classes)

    checkpoint_path = os.path.join(args.model_output_dir, "best_model.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(data, one_hot_labels, epochs=args.epochs, validation_split=0.1, callbacks=[checkpoint])

    training_time = time.time() - start_training_time
    print("Training time:", training_time)

    start_eval_time = time.time()

    final_model_path = os.path.join(args.model_output_dir, "final_model.h5")
    model.save(final_model_path)

    best_model = models.load_model(checkpoint_path)
    predictions = best_model.predict(data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(one_hot_labels, axis=1)

    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f'Accuracy: {accuracy}')

    confusion = confusion_matrix(true_classes, predicted_classes)
    print('Confusion Matrix:') 
    print(confusion)

    # Compute TP, FP, TN, FN from confusion matrix
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    FN = confusion.sum(axis=1) - TP
    TN = confusion.sum() - (TP + FP + FN)

    print(f'True Positives: {TP}')
    print(f'False Positives: {FP}')
    print(f'True Negatives: {TN}')
    print(f'False Negatives: {FN}')

    evaluation_time = time.time() - start_eval_time
    print("Evaluation time:", evaluation_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_output_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args, _ = parser.parse_known_args()
    
    main()
