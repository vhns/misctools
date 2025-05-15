#!/usr/bin/env python3

import os
import argparse
import csv
import tensorflow as tf
import numpy as np


def generator(csv_file, random):

    data = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file, delimiter=' ')
        for img in reader:
            data.append(img)

    idx = np.arange(len(data))

    if random:
        np.random.shuffle(idx)

    for img in idx:
        img = tf.keras.utils.load_img(img)
        img = tf.keras.utils.img_to_array(img)

        yield img, img


def calcular_camadas(filters_list=[8,16,32,64,128]):
    num_layers = np.random.randint(self.min_layers, self.max_layers + 1) #+1 por conta do randint ser somente de (min - max-1)

    encoder_layers = []
    decoder_layers = []
    maxpoll_layers = []
    batch = False
    leaky = False


    shape_atual = self.input_shape
    filter_sizes = [] 

    layers = {}

    # Encoder
    for i in range(num_layers):
        filters = np.random.choice(filters_list)
        filter_sizes.append(filters)
        encoder_layers.append(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        maxpoll_layers.append(np.random.choice([0, 0, 0, 1, 1, 1]))

        if len(maxpoll_layers) > 4:
            maxpoll_layers[i] = 0
        elif maxpoll_layers[i] == 1: #limitando o valor para 4, por conta de ser 64x64 
            encoder_layers.append(MaxPooling2D((2, 2), padding='same'))
            shape_atual = (shape_atual[0] // 2, shape_atual[1] // 2, filters) #att por conta da divisão do maxpool
        else:
            shape_atual = (shape_atual[0], shape_atual[1], filters) 

    latent_dim = np.random.randint(256,512)

    if np.random.choice(([0,0,1])):
        encoder_layers.append(BatchNormalization()) 
        batch = True
    if np.random.choice(([0,0,1])):
        encoder_layers.append(LeakyReLU(alpha=0.5)) #Função de ativação 
        leaky = True
    if np.random.choice(([0,0,1])):
        encoder_layers.append(Dropout(np.random.choice(([0.4 , 0.3 , 0.2])))) 

    encoder_layers.append(Flatten())

    encoder_layers.append(Dense(latent_dim, activation='relu')) #transformo o meu shape_atual no meu latent_dim 

    # Decoder
    decoder_layers = [
        Dense(np.prod(shape_atual), activation='relu'), #calcula o vetor latente 
        Reshape(shape_atual)
    ]

    for i in range(num_layers):
        filters = filter_sizes[-(i+1)]
        if maxpoll_layers[-(i+1)] == 1:
            decoder_layers.append(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        else:
            decoder_layers.append(Conv2DTranspose(filters, (3, 3), strides=(1, 1), padding='same', activation='relu'))

    if batch:
        decoder_layers.append(BatchNormalization())  
    if leaky:
        decoder_layers.append(LeakyReLU(alpha=0.5))

    decoder_layers.append(Conv2D(self.input_shape[2], (3, 3), activation='sigmoid', padding='same'))

    gc.collect()
    k.clear_session()

    return encoder_layers, decoder_layers, latent_dim

def construir_modelo(self, salvar=False, filters_list=[8,16,32,64,128]):
    # Limpar antigas referências 
    self.encoder = None
    self.decoder = None
    self.autoencoder = None


    encoder_layers, decoder_layers, latent_dim = self.calcular_camadas(filters_list)

    # Construir encoder
    inputs = Input(shape=self.input_shape)
    x = inputs
    for layer in encoder_layers:
        x = layer(x) #as camadas vão sequencialmente sendo adicionadas 
        self.encoder = Model(inputs, x, name='encoder')

    # Construir decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs
    for layer in decoder_layers:
        x = layer(x)
        print(x)
        self.decoder = Model(latent_inputs, x, name='decoder')

    # Construir autoencoder
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    self.autoencoder = Model(inputs, decoded, name=f'autoencoder')

    if salvar == True and self.nome_modelo !=None:
        save_dir = os.path.join(path, "Modelos", self.nome_modelo)
        dir_raiz = os.path.join( save_dir, "Modelo-Base")
        dir_modelo = os.path.join(dir_raiz, "Estrutura")
        dir_pesos = os.path.join(dir_raiz, "Pesos")

        recria_diretorio(save_dir)
        recria_diretorio(dir_raiz)
        recria_diretorio(dir_modelo)
        recria_diretorio(dir_pesos)

        self.autoencoder.save(f"{dir_modelo}/{self.nome_modelo}.keras")

    return self.autoencoder


def model():

    model = tf.keras.Sequential([
        # Resize input to 224x224 (if not already this size)
        tf.keras.layers.Resizing(height=224, width=224),
        
        # Encoder part
        tf.keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Bottleneck (reshape to a 4x4x4 tensor for the decoder)
        tf.keras.layers.Reshape((4, 4, 4)),
        
        # Decoder part (transposed convolutions to upsample)
        tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        
        # Output layer: 224x224x3 image with sigmoid activation
        tf.keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')
    ])

    model.compile(optimizer='Adam', loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--csv', required=True, type=str,
                        help='Path to the file inputs CSV')
    parser.add_argument('--output', required=False, type=str, default=
                        help='Path for the model results')
    parser.add_argument('--epochs', required=False, type=int, default='50',
                        help='Amount of epochs to train the model')
    parser.add_argument('--random', required=False, type=bool, default='False',
                        help='Whether we should randomize the items in the input.')
    parser.add_argument('--size', required=False, type=str, default='192,192')

    args = parser.parse_args()

    model = model()

    output_signature = (tf.TensorSpec(shape=(200, 200, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(200, 200, 3), dtype=tf.float32))
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        args=[args.csv, args.size, args.random],
        output_signature=output_signature)

    callback = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{args.output}/best.weights.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.CSVLogger(
            filename=str(f'{args.output}/training.log'),
            separator=',',
            append=False)
    ]

    os.makedirs(args.output, exist_ok=True)

    history = model.fit(dataset.batch(8).prefetch(4), epochs=args.epochs,
                        callbacks=callback)
    
