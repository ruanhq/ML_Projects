#############Core Functions for LSTM-Text-classification##########


import nltk
import sklearn
import os
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_auc_score, roc_curve, 
import numpy as np
from lxml import html, etree
import keras
from keras.layers import Lambda
from keras import backend as K
import xgboost as xgb
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Activation, Embedding, Dropout
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
#nltk.download()
from keras.models import Sequential, Graph
# Download Corpora -> stopwords, Models -> punkt
from keras import losses
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from lxml import html, etree
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from lxml import html, etree
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Activation, Embedding, Dropout, Input, RepeatVector, TimeDistributed, Bidirectional
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


#######
#Simple LSTM for binary classification:
def LSTM_simple_binary():
    model_ls1 = Sequential()
    model_ls1.add(Embedding(MAX_VOC, embedding_dim, input_length = max_length,
                           weights = [embedding_matrix],
                           trainable = False))
    model_ls1.add(LSTM(64))
    model_ls1.add(Dense(1, activation = 'sigmoid'))
    model_ls1.add(Dropout(0.2))
    OPTIM = Adam(lr=0.002)
    model_ls1.compile(loss = 'binary_crossentropy',
                 optimizer = OPTIM, metrics = ['accuracy'])
    model_ls1.summary()
    return model_ls1

#Multi-classification:
def LSTM_simple_multi():
    model_ls1 = Sequential()
    model_ls1.add(Embedding(vocab_size, embedding_dim, input_length = max_length,
                           weights = [embedding_matrix],
                           trainable = False))
    model_ls1.add(LSTM(64))
    model_ls1.add(Dense(20, activation = 'softmax'))
    model_ls1.add(Dropout(0.2))
    OPTIM = Adam(lr=0.002)
    model_ls1.compile(loss = 'categorical_crossentropy',
                 optimizer = OPTIM, metrics = ['accuracy'])
    model_ls1.summary()
    return model_ls1



##########
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
    OPTIM = Adam(lr = 0.005)

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


    autoencoder = Model(input = X_input, outputs = [output, pred])
    autoencoder.compile(optimizer = OPTIM, loss =  {'AE':'mse', 'CLASF':'binary_crossentropy'}, metrics = ['accuracy'])

    return autoencoder



#JOintly train an autoencoder and a classifier:
def LSTM_AE_clf_multi(n_feature,
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
    OPTIM = Adam(lr = 0.005)

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


    #Attach a classifier at the latent space:
    h1 = Dense(np.int(latent_dim / 2), activation = 'relu')(latent)
    h1 = Dropout(0.2)(h1)
    pred = Dense(20, activation = 'softmax', name = 'CLASF')(h1)


    autoencoder = Model(input = X_input, outputs = [output, pred])
    autoencoder.compile(optimizer = OPTIM, loss =  {'AE':'mse', 'CLASF':'binary_crossentropy'}, metrics = ['accuracy'])

    return encoder, decoder, autoencoder



#Bidirectional-LSTM for binary classifier:
def LSTM_simple_bi():
    model_ls1 = Sequential()
    model_ls1.add(Embedding(2000, output_dim = 64))
    model_ls1.add(Bidirectional(LSTM(48, return_sequences = True)))
    model_ls1.add(Bidirectional(LSTM(48)))
    model_ls1.add(Dense(1, activation = 'sigmoid'))
    OPTIM = Adam(lr=0.001)
    model_ls1.compile(loss = 'binary_crossentropy',
                 optimizer = OPTIM, metrics = ['accuracy'])
    model_ls1.summary()
    return model_ls1



#Bidirectional-LSTM for binary classifier:
def LSTM_simple_bi_multi():
    model_ls1 = Sequential()
    model_ls1.add(Embedding(vocab_size, embedding_dim, input_length = max_length,
                           weights = [embedding_matrix],
                           trainable = False))
    model_ls1.add(Bidirectional(LSTM(48, return_sequences = True)))
    model_ls1.add(Bidirectional(LSTM(48)))
    model_ls1.add(Dense(20, activation = 'softmax'))
    model_ls1.add(Dropout(0.2))
    OPTIM = Adam(lr=0.001)
    model_ls1.compile(loss = 'categorical_crossentropy',
                 optimizer = OPTIM, metrics = ['accuracy'])
    model_ls1.summary()
    return model_ls1



#Joint training VAE and classifier onto the 
def LSTM_VAE_clf_binary(n_feature,
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


    return VAE_clf



###########
#Joint VAE-LSTM:
def LSTM_VAE_clf_multi(n_feature,
    inter_dim1,
    latent_dim,
    vocab_size,
    embedding_dim,
    max_length,
    embedding_matrix):
    """
    :Implement the VAE algorithm to do classification with the LSTM as the component architecture and jointly train a autoencoder and a classifier simultaneously.
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
    I = Dense(40, activation = 'relu')(z)
    I = Dropout(0.2)(I)
    pred = Dense(20, activation = 'softmax', name = 'CLASF')(I)

    VAE_clf = Model(input = X_input, outputs = [decoded, pred])
    VAE_clf.compile(optimizer = OPTIM, loss = {'VAE' : vae_loss, 'CLASF':'categorical_crossentropy'}, metrics = ['accuracy'])


    return VAE_clf



#####
#Kappa statistic evaluation:
def kappa_stats(confusion_matrix):
    #Calculate the Cochren's kappa for a binary classification problem
    #Input is a confusion matrix for a binary classification problem:
    tp, fp, fn, tn = confusion_matrix[0,0], confusion_matrix[0,1], confusion_matrix[1,0], confusion_matrix[1,1] 
    po = (tp + tn)/(tp + fp + tn + fn)
    pe = (tp + fn) * (tp + fp) / ((confusion_matrix.sum())**2)
    kappa = (po - pe) / (1 - pe)
    return kappa


######
#Calculate the statistic for evaluation
def cal_stats_ncategory(confusion_matrix, n):
    #Calculate the statistics for an arbitrary confusion matrix:
    """
    Input: 
    confusion_matrix denotes an arbitrary confusion matrix(array)
    n denotes the number of categories
    Output:
    Micro-FPR, Micro-TPR, Micro-Precision, Micro-Recall, Kappa statistics
    Macro version for the previous ones

    """
    fpr_dic = []
    fOr_dic = []
    pre_dic = []
    rec_dic = []
    acc_dic = []
    kappa_dic = []
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    #Enumerate for each pair of predictors.
    for i in range(n):
        for j in range(i, n):
            if i != j:
                curr_conf_mat =  np.array([[confusion_matrix[i,i], confusion_matrix[i,j]],[confusion_matrix[j, i], confusion_matrix[j,j]]])
                curr_fpr = 1 - curr_conf_mat[1,1]/(curr_conf_mat[1,1] + curr_conf_mat[0,1])
                curr_fOr = 1 - curr_conf_mat[1,1]/(curr_conf_mat[1,1] + curr_conf_mat[1,0])
                curr_pre = curr_conf_mat[0,0]/(curr_conf_mat[0,0] + curr_conf_mat[0,1])
                curr_rec = curr_conf_mat[0,0]/(curr_conf_mat[0,0] + curr_conf_mat[1,0])
                curr_acc = np.diag(curr_conf_mat).sum()/curr_conf_mat.sum()
                curr_kappa = kappa_stats(curr_conf_mat)
                fpr_dic.append(curr_fpr)
                fOr_dic.append(curr_fOr)
                pre_dic.append(curr_pre)
                rec_dic.append(curr_rec)
                acc_dic.append(curr_acc)
                kappa_dic.append(curr_kappa)
                tp += curr_conf_mat[0,0]
                tn += curr_conf_mat[1,1]
                fp += curr_conf_mat[0,1]
                fn += curr_conf_mat[1,0]

    ##Calculate the micro statistics(from pairwise predictors):
    micro_fpr = np.mean(fpr_dic)
    micro_fOr = np.mean(fOr_dic)
    micro_pre = np.mean(pre_dic)
    micro_rec = np.mean(rec_dic)
    micro_acc = np.mean(acc_dic)
    micro_kappa = np.mean(kappa_dic)

    #Calculate the macro statistics(cumulation-wise predictors)
    macro_pre = tp/(tp + fp)
    macro_rec = tp/(tp + fn)
    macro_fpr = 1 - tn/(tn + fp)
    macro_fOr = 1 - tn/(tn + fn)
    macro_acc = (tp + fn) / (tp + fn + fp + tn)
    mcc = (tp * tn - fp * fn)/np.sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp))
    macro_kappa = kappa_stats(np.array([[tp, fp], [fn, tn]]))

    #Create a dataframe for the classification report:
    classification_report = pd.DataFrame({'mi_acc':[micro_acc], 'mi_fpr': [micro_fpr], 'mi_fOr': [micro_fOr], 'mi_rec': [micro_rec], 'mi_pre': [micro_pre],
        'mi_kappa': [micro_kappa], 'ma_acc':[macro_acc], 'ma_fpr': [macro_fpr], 'ma_fOr': [macro_fOr], 'ma_rec': [macro_rec], 'ma_pre': [macro_pre],
        'ma_kappa': [macro_kappa], 'MCC': [mcc]})

    return classification_report



############
#Function to calculate the corresponding statistics for the classifier:
def cal_score(Y_label, scoring):
    #AUROC/ AUPR calculation:
    fpr, tpr, _ = roc_curve(Y_label, scoring)
    auroc = roc_auc_score(Y_label, scoring)
    pr, rec, _ = precision_recall_curve(Y_label, scoring)
    aupr = average_precision_score(Y_label, scoring)
    #####
    plt.step(fpr, tpr, color = 'blue')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('AUC is {:.6f}'.format(auroc))
    plt.show()
    
    plt.step(rec, pr, color = 'b')
    plt.xlabel('PR')
    plt.ylabel('REC')
    plt.title('AUPR is {:.6f}'.format(aupr))
    
    #Transform the scoring to binary prediction:
    label_pred = [1 if t > 0.5 else 0 for t in scoring]
    
    #Output confusion matrix:
    conf_mat = confusion_matrix(Y_label, label_pred)
    
    #Evaluation the for label classification:
    accu = accuracy_score(Y_label, label_pred)
    prec = precision_score(Y_label, label_pred)
    recc = recall_score(Y_label, label_pred)
    kap = cohen_kappa_score(Y_label, label_pred)
    
    #Generate report:
    report = pd.DataFrame({'auc': [auroc], 'aupr': [aupr], 'accu': [accu], 'prec': [prec], 'recc': [recc], 'kap': [kap]})
    
    return report, conf_mat
    


#Calculate the statistics for each of the model:
def cal_stats(Y_label, Y_pred):
    #First transform the one-hot encoding to label encoding:
    
    Y_test = [np.argmax(t) for t in Y_label]
    
    #Extract the confusion-matrix:
    conf_mat = confusion_matrix(Y_test, Y_pred)
    
    #Get the mean precision/recall for multi-label classification:
    prec = precision_score(Y_test, Y_pred, average = 'macro')
    recc = recall_score(Y_test, Y_pred, average = 'macro')
    
    #Calculate the kappa statistic:
    kap = cohen_kappa_score(Y_test, Y_pred)
    
    #Calculate the accuracy:
    acc = accuracy_score(Y_test, Y_pred)
    
    #Create a dataframe for summary:
    report = pd.DataFrame({'ACC':[acc], 'precision': [prec], 'recall': [recc], 'Kappa': [kap]})
    
    return report



