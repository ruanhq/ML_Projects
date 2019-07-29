#VAE evaluation:
def LSTM_VAE_clf(n_feature,
	inter_dim1,
	latent_dim,
	vocab_size,
	embedding_dim,
	max_length,
	embedding_matrix):
	"""
	:Implement the VAE algorithm to do classification with the LSTM as the component architecture
	:::
	Params:
	n_feature        : # of input features
	inter_dim1       : # of hidden units
	latent_dim       : dimension of latent space
	vocab_size       : Vocabulary size
    embedding_dim    : The dimension of word embedding
    max_length       : The maximum length of words we use
    embedding_matrix : Pre-specified word embedding matrix
    """
	OPTIM = Adam(lr = 0.005)

	#Embedding: Start from word count matrix:
	X_input = Input(shape = (n_feature, ))
	X = Embedding(vocab_size, embedding_dim, input_length = max_length,
		weights = [embedding_matrix], trainable = False)(X_input)


	#Encoder:
	ENC = LSTM(inter_dim1, return_sequences = False)(X)
	ENC = Dense(inter_dim1, activation = 'relu')(ENC)

	def sampling(args):
		z_mean_, z_log_sigma_ = args
		batch_size = K.shape(z_mean_)[0]
		epsilon = K.random_normal(shape = (batch_size, latent_dim), mean = 0,
			stddev = 1.0)
		z = z_mean_ + K.exp(z_log_sigma_ / 2) * epsilon
		return z

	z_mean = Dense(latent_dim, activation = 'relu')(ENC)
	z_log_sigma = Dense(latent_dim, activation = 'relu')(ENC)

	z = Lambda(sampling, output_shape = (latent_dim, ))([z_mean, z_log_sigma])


	#Decoder:
	DEC = RepeatVector(embedding_dim)(z)

	J = LSTM(inter_dim1, return_sequences = True)(DEC)
	DEC = LSTM(inter_dim1)(DEC)

	decoded = Dense(n_feature, activation = 'softmax', name = 'VAE')(DEC)

	#Define VAE loss:
	def vae_loss(x, x_dec):
		#recon_loss + kl_loss
		recon_loss = losses.mean_squared_error(x, x_dec)
		kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = -1)
		vae_loss = recon_loss + kl_loss

		return vae_loss

	#Attach another classifier at the latent space  of the VAE:
	I = Dense(np.int(latent_dim / 2), activation = 'relu')(z)
	I = Dropout(0.2)(I)
	pred = Dense(1, activation = 'sigmoid', name = 'CLASF')(I)



	VAE_clf = Model(input = X_input, outputs = [decoded, pred])
	VAE_clf.compile(optimizer = OPTIM, loss = {'VAE' : vae_loss, 'CLASF':'binary_crossentropy'}, metrics = ['accuracy'])
	encoder = Model(input = X_input, outputs = z)

	return encoder, VAE_clf