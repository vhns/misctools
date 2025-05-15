#!/usr/bin/env python3

import os
import csv
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

def generator(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file, delimiter=' ')
        for i in reader:
            filename = i[0]
            img = tf.keras.utils.load_img(filename)
            img = tf.keras.utils.img_to_array(img)
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

            yield img, filename

def gen_model():
    image_inputs = tf.keras.Input(shape=(385, 387, 3))
    model_resize = tf.keras.layers.Resizing(height=224, width=224)(image_inputs)
    mobilenet = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                  include_top=False,
                                                  weights='imagenet',
                                                  pooling='avg')(model_resize)
    model = tf.keras.Model(inputs=image_inputs, outputs=mobilenet, name='simple_model')

    for layer in model.layers[2].layers:
        if 'block_15' in layer.name or 'block_16' in layer.name:
            layer.trainable = True
            print(layer)
        else:
            layer.trainable = False

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def run_model(model, output_path, csv):

    files = iter(generator(csv))

    for i in files:
        img, filename = i
        img = np.expand_dims(img, axis=0)
        result = model.predict(img)
        filename = Path(filename).stem
        filename = f'{filename}.py'
        filename = os.path.join(output_path, filename)
        np.save(filename, result)

if __name__ == '__main__':
    args = argparse.ArgumentParser(
            prog='Trains a simple model on spectrograms',
            description='Does what it says on the tin according to given parameters')

    args.add_argument('--output-path', type=str, required=True)
    args.add_argument('--csv', type=str, required=True)

    args = args.parse_args()

    model = gen_model()

    os.makedirs(args.output_path, exist_ok=True)

    run_model(model, args.output_path, args.csv)
