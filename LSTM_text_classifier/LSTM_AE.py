#Jointly train an autoencoder and a classifier:
def LSTM_AE_clf_binary(n_feature,
                inter_dim1, 
                latent_dim, 
                vocab_size, 
                embedding_dim, 
                max_length, 
                embedding_matrix):

	"""
	:Construct a variational autoencoder with LSTM as component.
	Param:
	input_dim1:
	input_dim2:
	inter_dim1: Intermediate dimension for hidden layer
	latent_dim: 
	"""
	OPTIM = Adam(lr = 0.002)

	#Embedding: Start from the word count frequency matrix and first we get the word embedding from a previous text database:
	X_input = Input(shape = (n_feature, ))
	X = Embedding(vocab_size, embedding_dim, input_length = max_length,
		weights = [embedding_matrix], trainable = False)(X_input)

	#Encoder:
	ENC = LSTM(inter_dim1, return_sequences = True)(X)

	latent = LSTM(latent_dim)(ENC)

	#Decoder:
	DEC = RepeatVector(embedding_dim)(latent)
	DEC = LSTM(inter_dim1, return_sequences = True)(DEC)
	DEC = LSTM(inter_dim1)(DEC)
	output = Dense(embedding_dim, activation = 'softmax', name = 'AE')(DEC)


	#Attach a classifier:
	h1 = Dense(np.int(latent_dim / 2), activation = 'relu')(latent)
	h1 = Dropout(0.2)(h1)
	pred = Dense(1, activation = 'sigmoid', name = 'CLASF')(h1)
	#
	encoder = Model(input = X_input, outputs = latent)
	autoencoder = Model(input = X_input, outputs = [output, pred])
	autoencoder.compile(optimizer = OPTIM, loss =  {'AE':'mse', 'CLASF':'binary_crossentropy'}, metrics = ['accuracy'])

	return encoder, autoencoder

