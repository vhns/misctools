#!/usr/bin/env python3

import os
import argparse
import time
import eer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from csv import reader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def input_generator(csv_file, img_size, load_img=True):
    with open(csv_file, mode='r') as f:
        data = f.readlines()
        
    data = reader(data, delimiter=' ')
    data = list(data)

    if load_img:        
        for img, label in data:

            # TODO: Possibly augment these images?
            img = tf.keras.utils.load_img(img)
            img = img.resize((img_size, img_size))
            img = tf.keras.utils.img_to_array(img)/255.0

            yield img, label
    else:
        for _, label in data:
            yield label


def gen_model(model_path, weights_path=None, trainable):

    # We need to load the model, get it's config and generate a model
    # from the config, as to avoid possibly loading weights.
    model_inherit = tf.keras.models.load_model(model_path)
    config = model_inherit.get_config()
    model_inherit = tf.keras.models.from_config(config)
    config = None

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
        model_inherit = tf.keras.models.Model(inputs=model_inherit.input,
                                              outputs=model_inherit.layers[layer_index-1].output)
        print(f"Layer '{layer_to_remove}' removed successfully!")
    else:
        print(f"Layer '{layer_to_remove}' not found in the model.")

    model_inherit.trainable = trainable

    model = tf.keras.Sequential([
        model_inherit,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')])
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if weights_path:
        model.load_weights(weights_path, skip_mismatch=True)

    return model


def gen_dataset(csv_file, img_size):
    output_signature = (tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.uint8))

    return tf.data.Dataset.from_generator(input_generator,
                                          args=[csv_file, img_size],
                                          output_signature=output_signature)



def plot_train_val_acc(history, output_path):
    # Extract accuracy values from the history
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Extract epoch numbers
    epochs = range(1, len(train_accuracy) + 1)

    # Plot the training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, 'acc.svg'), format='svg')
    plt.close()

def plot_train_val_loss(history, output_path):
    # Extract  loss values from the history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Extract epoch numbers
    epochs = range(1, len(train_loss) + 1)

    # Plot the training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')

    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, 'loss.svg'), format='svg')
    plt.close()

def plot_cm_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_path, 'cm.svg'), format='svg')
    plt.close()

def gen_f1_report(y_true, y_pred, output_path):
    report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])

    with open(os.path.join(output_path, 'f1.txt'), mode='x') as f:
        print(report, file=f)

if __name__ == '__main__':
    # TODO: possibly store the current seed, save it (by generating it) as well as accept
    # it as an input parameter.
    args = argparse.ArgumentParser(
        prog='Trains a model on spectrograms, transfer-learning weights.',
        description='Does what it says on the tin according to given parameters.')

    args.add_argument('--output-path', type=str, default=time.strftime('%G %H:%M:%S'),
                      required=False)
    args.add_argument('--train-csv', type=str, required=True)
    args.add_argument('--test-csv', type=str, required=True)
    args.add_argument('--val-csv', type=str, required=True)
    args.add_argument('--epochs', type=int, default=50, required=False)
    args.add_argument('--weights-path', type=str, required=False)
    args.add_argument('--model-path', type=str, required=True)
    args.add_argument('--img-size', type=int, default=64, required=False)
    args.add_argument('--trainable', type=int, required=True) # TODO: figure out if Python
                                                              # has a smart way to get bool
    args = args.parse_args()                                  # arguments.

    # TODO: avoid defining paths with things such as os.path.join(args.foobar, 'baz') and
    # instead, define them all here at the beginning.

    os.makedirs(args.output_path, exist_ok=False)

    # TODO:
    # As our image generator is not native to Tensorflow, it is not able to predict
    # how many steps each epoch will have. As a result, unless we inform it of such
    # info, it'll return INFO messages. Also, as // floors the result, we end up
    # needing to pass a .repeat() later on to the datasets. This *needs* to be fixed.
    # The reason for that is because say we have data as below:
    # [0,1,2,3,4,5,6,7,8,9]
    # and we are running at a batch of 4 and do the division as below, resulting in
    # 2 steps per epoch. Our generator will go up to item 7 and finish training for
    # that epoch. On the next epoch, it will start training at item 8 and end at item
    # 5. As you can see, not only do we not have a predictable training process, some
    # items may be seen more often than others, even if by just a small margin.
    train_ds = gen_dataset(args.train_csv, args.img_size)
    train_steps = len(open(args.train_csv,mode='r').readlines()) // 32 # TODO: use an argument for
    test_ds = gen_dataset(args.test_csv, args.img_size)                # the number of batches.
    test_steps = len(open(args.test_csv,mode='r').readlines()) // 32    
    val_ds = gen_dataset(args.val_csv, args.img_size)
    val_steps = len(open(args.val_csv,mode='r').readlines()) // 32
    
    model = gen_model(args.model_path, args.weights_path, bool(args.trainable))
    model_train_log = os.path.join(args.output_path, 'model_train.log')
    model_test_log = os.path.join(args.output_path, 'model_test.log')

    initial_time = time.time()

    print(f'{time.strftime("%G %H:%M:%S", time.localtime(initial_time))}: Model training started.')

    callbacks = [ tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.output_path, '{epoch:02d}.weights.h5'),
        monitor='val_loss',
        mode='max',
        save_best_only=False,
        save_weights_only=True),
                  tf.keras.callbacks.CSVLogger(
                      filename=model_train_log,
                      separator=',',
                      append=True)
                 ]

    history = model.fit(x=train_ds.batch(32).prefetch(4).repeat(args.epochs),
                        epochs=args.epochs,
                        validation_data=val_ds.batch(32).prefetch(4).repeat(args.epochs),
                        callbacks=callbacks,
                        steps_per_epoch=train_steps,
                        validation_steps=val_steps)
    
    final_time = time.time()
    
    print(f'{time.strftime("%G %H:%M:%S", time.localtime(final_time))}:',
          'Model\'s training concluded.')

    initial_time = time.time()
    
    print(f'{time.strftime("%G %H:%M:%S", time.localtime(initial_time))}:',
          'Model testing started.')

    callbacks = [ tf.keras.callbacks.CSVLogger(
                      filename=model_test_log,
                      separator=',',
                      append=True)
                 ]

    predictions = model.predict(x=test_ds.batch(32).prefetch(4),
                                callbacks=callbacks,
                                steps=test_steps)

    final_time = time.time()

    print(f'{time.strftime("%G %H:%M:%S", time.localtime(final_time))}:',
          'Model\'s testing concluded.')
        
    y_true = [int(i) for _, i in iter(test_ds)]
    y_true = y_true[:(test_steps*32)] # TODO: Replace number with argument of batches.
    y_pred = np.argmax(predictions, axis=1)

    plot_train_val_acc(history, args.output_path)
    plot_train_val_loss(history, args.output_path)
    plot_cm_matrix(y_true, y_pred, args.output_path)
    gen_f1_report(y_true, y_pred, args.output_path)

    np.save(os.path.join(args.output_path, 'y_true.npy'), y_true)
    np.save(os.path.join(args.output_path, 'y_pred.npy'), y_pred)

    y_scores = predictions[:,1]

    out_csv = os.path.join(args.output_path, 'eer.csv')
    out_svg = os.path.join(args.output_path, 'eer.svg')
    
    eer.generate_eer_report(y_true, y_scores, out_csv, out_svg)
    eer_threshold, eer_value = eer.calculate_eer(y_true, y_scores)
    foo_out = os.path.join(args.output_path, 'eer.txt')
    with open(foo_out, mode='x') as f:
        print(f'Threshold: {eer_threshold:.2f} Value: {eer_value*100:.2f}%', file=f)
