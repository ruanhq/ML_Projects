import sklearn
import numpy as np
from sklearn.manifold import TSNE
import os
import keras
from keras.layers import Layer, Dropout, LeakyReLU
from keras.layers import LSTM, Dense, Activation, Embedding, Dropout, Input, RepeatVector, TimeDistributed, Bidirectional

#Genomic data application on T4S1 dataset classification:


AA_T4 = np.asarray(pd.read_csv('T4_scRNA.csv', sep = ' '))
BB_S1 = np.asarray(pd.read_csv('S1_scRNA.csv', sep = ' '))
AB_T4S1 = np.asarray(pd.read_csv('T4S1_scRNA.csv', sep = ' '))
n_A = AA_T4.shape[0]
n_B = BB_S1.shape[0]
labels_t4s1= np.zeros((n_A + n_B ,2))
labels_t4s1[:n_A,0] = 1
labels_t4s1[n_A:(n_A + n_B), 1] = 1


#Global constants:
N = AB_T4S1.shape[0]
F = AB_T4S1.shape[1]
F_ = 6
n_attn_heads = 6
dropout_rate = 0.4
l2_reg = 1e-3
learning_rate = 1e-3
n_class = 2
NN_k = 50

#Simulate KNN structure on the scRNA data:
AB_T4S1 = np.concatenate([AA_T4, BB_S1])
KNN_T4S1 = kneighbors_graph(AB_T4S1, n_neighbors = NN_k)

#Subsampling the KNN structure on the scRNA data:
import networkx as nx
KNN_subs_T4S1 = nx.


#Construct the GAT classifier model:
X_in = Input(shape = (F, ))
A_in = Input(shape = (N, ))
n_classes = 2
dr1 = Dropout(dropout_rate)(X_in)
#First GAT layer:
gat_1, attn1 = GATLayer(F_, num_attn_head = n_attn_heads,
	dropout_rate = dropout_rate,
	activation = 'elu',
	kernel_regularizer = l2(l2_reg))([dr1, A_in])
#Dense layer:
AS1 = Dense(32, activation = 'relu')(gat_1)

#Second GAT layer:
gat_2, attn2 = GATLayer(2, num_attn_head = 1,
	attn_head_reduction = 'average',
	dropout_rate = dropout_rate,
	activation = 'elu',
	kernel_regularizer = l2(l2_reg))([AS1, A_in])

#Learned Labels via softmax activation:
learned_labels = Dense(n_classes, activation = 'softmax')(gat_2)

#Building model:
model = Model(inputs = [X_in, A_in], outputs = learned_labels)
#Extract the low dimensional manifold:
sub_model1 = Model(inputs = [X_in, A_in], outputs = gat_2)
#Extract the attention coefficient for each of the model:
sub_model2 = Model(inputs = [X_in, A_in], outputs = attn2)

#Call-back selecting the best model:
Model_callback = ModelCheckpoint('logs/best_model.h5',
	monitor = 'weighted_acc',
	save_best_only = True,
	save_weights_only = True)

#Fit the model:
optimizer = Adam(lr = learning_rate)
model.compile(optimizer = optimizer,
	loss = 'binary_crossentropy',
	metrics = ['acc'])
model.summary()
history1 = model.fit([AB_T4S1, KNN_T4S1],
	labels_t4s1,
	epochs = 30,
	batch_size = N,
	shuffle = False,
	callbacks = Model_callback)

low_dim_rep = sub_model1.predict([AB_T4S1, KNN_T4S1], batch_size = N)

#Extract the attention coefficients:
attn_coef = sub_model2.predict([AB_T4S1, KNN_T4S1], batch_size = N)

#Visualize the low dimensional embedding via TSNE:
tsne = TSNE(n_components = 2, perplexity = 35)
tsne_2 = tsne.fit_transform(AB_T4S1)

#Visualize the near the boundary of manifold:


