import os
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
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

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_output_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args, _ = parser.parse_known_args()
    
    # Load dataset from the input directory
    input_data_path = os.environ.get('SM_CHANNEL_TRAIN', 'FaceDisguiseDatabase/FaceAll')
    
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

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_labels)

    model = build_and_compile_model((64, 64, 3), len(np.unique(encoded_labels)))

    checkpoint_path = os.path.join(args.model_output_dir, "best_model.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(data, one_hot_labels, epochs=args.epochs, validation_split=0.1, callbacks=[checkpoint])

    final_model_path = os.path.join(args.model_output_dir, "final_model.h5")
    model.save(final_model_path)
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
    print("Total training time:", total_time)