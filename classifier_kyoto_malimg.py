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
            img = img.resize((64, 64), resample=1)# Talvez mudar o tamanho da imagem.
            # Possível não funcionar devido aos pesos
            img = tf.keras.utils.img_to_array(img)/255.0
            # img = tf.keras.utils.img_to_array(img)

            yield img, label

def gen_model(model_path, weights_path):

    model_inherit = tf.keras.models.load_model(model_path)
    # model_inherit = model_inherit.get_layer('encoder')

    # Specify the name of the layer to remove
    layer_to_remove = 'decoder'

    # Find the index of the layer with the specified name
    layer_index = None
    for i, layer in enumerate(model_inherit.layers):
        if layer.name == layer_to_remove:
            layer_index = i
            break

    if layer_index is not None:
        # Create a new model without the layer
        model_inherit = tf.keras.models.Model(inputs=model_inherit.input, outputs=model_inherit.layers[layer_index-1].output)
        print(f"Layer '{layer_to_remove}' removed successfully!")
    else:
        print(f"Layer '{layer_to_remove}' not found in the model.")

    model = tf.keras.Sequential([
        model_inherit,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(24, activation='softmax')])
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.load_weights(weights_path, skip_mismatch=True)
    
    return model


def gen_dataset(generator, csv_file):
    output_signature = (tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.uint8))

    return tf.data.Dataset.from_generator(generator,
                                          args=[csv_file],
                                          output_signature=output_signature)


def graph_data(history, output_path):

    os.makedirs(output_path, exist_ok=True)
   
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

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, 'loss.png'))
    plt.close()


def graph_confusion(model, output_path):

    os.makedirs(output_path, exist_ok=True)

    y_true = []

    for _, label in test_ds.batch(32):
        y_true.extend(label.numpy())

    y_pred = model.predict(test_ds.batch(32))
    y_pred = np.argmax(y_pred, axis=1)

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
    

def run_model(model, output_path, train_dataset, test_dataset, val_dataset, epochs):

    callbacks = [ tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            output_path, 'best.weights.h5'),
        monitor='val_loss',
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
                        callbacks=callbacks,
                        verbose=2)

    model.save_weights(os.path.join(output_path, 'final.weights.h5'))

    graph_data(history, os.path.join(output_path, 'final'))
    graph_confusion(model, os.path.join(output_path, 'final'))
    
    model.load_weights(os.path.join(output_path, 'best.weights.h5'))
    graph_data(history, os.path.join(output_path, 'best'))
    graph_confusion(model, os.path.join(output_path, 'best'))


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        prog='Trains a simple model on spectrograms',
        description='Does what it says on the tin according to given parameters')

    args.add_argument('--output-path', type=str, required=True)
    args.add_argument('--train-csv', type=str, required=True)
    args.add_argument('--test-csv', type=str, required=True)
    args.add_argument('--val-csv', type=str, required=True)
    args.add_argument('--epochs', type=int, required=True)
    args.add_argument('--model-path', type=str, required=True)
    args.add_argument('--model-weights', type=str, required=True)


    args = args.parse_args()

    train_ds = gen_dataset(input_generator, args.train_csv)
    test_ds = gen_dataset(input_generator, args.test_csv)
    val_ds = gen_dataset(input_generator, args.val_csv)

    model = gen_model(args.model_path, args.model_weights)

    os.makedirs(args.output_path, exist_ok=True)
    
    initial_time = time()

    run_model(model, args.output_path, train_ds, test_ds, val_ds, args.epochs)
    
    print(f'Elapsed time: {time() - initial_time}')
