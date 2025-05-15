#!/usr/bin/env python3

import os
import csv
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def input_generator(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file, delimiter=' ')
        for img, label in reader:
            img = tf.keras.utils.load_img(img)
            img = tf.keras.utils.img_to_array(img)
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

            yield img, label

def gen_model():
    image_inputs = tf.keras.Input(shape=(385, 387, 3))
    model_resize = tf.keras.layers.Resizing(height=224, width=224)(image_inputs)
    mobilenet = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                  include_top=False,
                                                  weights='imagenet',
                                                  pooling='avg')(model_resize)
    x = tf.keras.layers.Flatten()(mobilenet)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=image_inputs, outputs=x, name='simple_model')

    for layer in model.layers[2].layers:
        if 'block_15' in layer.name or 'block_16' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def gen_dataset(generator, csv_file):
    output_signature = (tf.TensorSpec(shape=(385, 387, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.uint8))

    return tf.data.Dataset.from_generator(generator,
                                          args=[csv_file],
                                          output_signature=output_signature)

def run_model(model, output_path, train_dataset, test_dataset, val_dataset, epochs):

    callbacks = [ tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            output_path, 'best.weights.h5'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True),
                  tf.keras.callbacks.CSVLogger(
                      filename=os.path.join(
                          output_path, 'model.log'),
                      separator=',',
                      append=False)
                 ]

    history = model.fit(x=train_dataset.batch(32).prefetch(4),
                        epochs=epochs,
                        validation_data=val_dataset.batch(32).prefetch(4),
                        callbacks=callbacks)

    model.save_weights(os.path.join(output_path, 'final.weights.h5'))

    # Extract accuracy values from the history
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history.get('val_accuracy', [])

    # Extract epoch numbers
    epochs = range(1, len(train_accuracy) + 1)

    # Plot the training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
    if val_accuracy:
        plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, 'acc.png'))
    plt.close()

    # Extract accuracy values from the history
    train_accuracy = history.history['loss']
    val_accuracy = history.history.get('val_loss', [])

    # Extract epoch numbers
    epochs = range(1, len(train_accuracy) + 1)

    # Plot the training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, label='Training Loss', marker='o')
    if val_accuracy:
        plt.plot(epochs, val_accuracy, label='Validation Loss', marker='o')

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, 'loss.png'))
    plt.close()

    y_pred = []
    y_true = []

    model.load_weights(os.path.join(output_path, 'best.weights.h5'))

    for x_batch, y_batch in test_dataset.batch(32):
        preds = model.predict(x_batch)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y_batch.numpy())

    cm = confusion_matrix(y_true, y_pred)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_path, 'cm.png'))
    plt.close()

    report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])

    with open(os.path.join(output_path, 'f1.txt'), mode='x') as f:
        print(report, file=f)


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        prog='Trains a simple model on spectrograms',
        description='Does what it says on the tin according to given parameters')

    args.add_argument('--output-path', type=str, required=True)
    args.add_argument('--train-csv', type=str, required=True)
    args.add_argument('--test-csv', type=str, required=True)
    args.add_argument('--val-csv', type=str, required=True)
    args.add_argument('--epochs', type=int, required=True)


    args = args.parse_args()

    train_ds = gen_dataset(input_generator, args.train_csv)
    test_ds = gen_dataset(input_generator, args.test_csv)
    val_ds = gen_dataset(input_generator, args.val_csv)

    model = gen_model()

    os.makedirs(args.output_path, exist_ok=True)
    
    initial_time = time()

    run_model(model, args.output_path, train_ds, test_ds, val_ds, args.epochs)
    
    print(f'Elapsed time: {time() - initial_time}')
