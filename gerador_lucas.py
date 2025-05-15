#!/usr/bin/env python3


class Gerador:
    """
    Dados de entrada: input_shape=(224, 224, 3), min_layers=2, max_layers=6\n
    """

    def __init__(self, input_shape=(224, 224, 3), min_layers=5, max_layers=8, nome_modelo:str=None):
        self.input_shape = input_shape
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.treino = None
        self.validacao = None
        self.teste = None
        self.nome_modelo = nome_modelo
        cria_pasta_modelos()

    def setNome(self, nome):
        self.nome_modelo = nome

    def getNome(self):
        print(self.nome)

    def getPesos(self):
        dir = f"weights_finais/Autoencoders_Gerados/{self.nome_modelo}.weights.h5"
        return str(dir)
    
    def calcular_camadas(self, filters_list=[8,16,32,64,128]):
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

    def compilar_modelo(self, optimizer='adam', loss='mse'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def treinar_autoencoder(self, salvar=False,nome_da_base='', epocas=10, batch_size=64):
        print("Treinando o modelo: ", self.nome_modelo)
        checkpoint_path = os.path.join(path, 'Pesos/Pesos_parciais/weights-improvement-{epoch:02d}-{val_loss:.2f}.weights.h5')
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                        save_weights_only=True, 
                                        monitor='val_loss', 
                                        mode='min', 
                                        save_best_only=True, 
                                        verbose=1)

        early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=150,  #interrompe se não melhorar
                               restore_best_weights=True, 
                               verbose=1)

        # Para caso querer acompanhar o treinamento
        history = self.autoencoder.fit(self.treino, epochs=epocas,callbacks=[cp_callback, early_stopping],batch_size=batch_size, validation_data=(self.validacao))
        df = pd.DataFrame(history.history)
        del history

        dir_pesos_save = os.path.join(path, 'Pesos/Pesos_parciais')
        if os.path.isdir(dir_pesos_save):
            shutil.rmtree(dir_pesos_save)

        if salvar == True and self.nome_modelo != None:
            save_dir = os.path.join(path, "Modelos", self.nome_modelo)
            dir_raiz = os.path.join(save_dir, "Modelo-Base")
            dir_modelo = os.path.join(dir_raiz, "Estrutura")
            dir_pesos = os.path.join(dir_raiz, "Pesos")
            dir_imagens = os.path.join(save_dir, "Plots")

            if os.listdir(dir_modelo) == []:
                recria_diretorio(dir_modelo)
                self.autoencoder.save(f"{dir_modelo}/{self.nome_modelo}.keras")

            recria_diretorio(dir_pesos)

            self.autoencoder.save_weights(f"{dir_pesos}/{self.nome_modelo}_Base-{nome_da_base}.weights.h5")

            if not os.path.isdir(dir_imagens):
                os.makedirs(dir_imagens)
            

        x, y = next(self.teste)
        plot_history(df, dir_imagens, self.nome_modelo, nome_da_base)
        plot_autoencoder(x, self.autoencoder, self.input_shape[0], self.input_shape[1],caminho_para_salvar=dir_imagens)

    def carrega_modelo(self, modelo:str, pesos:str=None):
        self.autoencoder = tf.keras.models.load_model(modelo)

        self.nome_modelo = retorna_nome(modelo)

        if pesos == False:
            print("Carregado somente a estrutura do modelo!")
        elif pesos !=None:  
            self.autoencoder.load_weights(pesos)
        

        self.decoder = self.autoencoder.get_layer('decoder')
        self.encoder = self.autoencoder.get_layer('encoder')

        self.autoencoder.summary()

        return self.autoencoder, self.encoder, self.decoder


    def predicao(self):
        x,y = next(self.teste)
        plot_autoencoder(x, self.autoencoder, self.input_shape[0],self.input_shape[1])
        pred = self.autoencoder.predict(x[0].reshape((1,self.input_shape[0], self.input_shape[1],3)))
        pred_img = normalize(pred[0])

        return x[0], pred_img
