#Functions library for performing EDA, modeling and feature engineering.

#Stacking average model predictions -> Mainly for regression.
class StackingAveragedModels_reg(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                #Here apply the second order model on the predicted features got by applying various models on each of the fold.
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

#Function to plot time series data:
def plot_ver_time(df, column, cal = 'mean', span = 15):
    if cal == 'mean':
        group_temp = df.groupby('date')[column].mean().reset_index()
    if cal == 'count':
        group_temp = df.groupby('date')[column].mean().reset_index()
    if cal == 'nunique':
        group_temp = df.groupby('date')[column].mean().reset_index()
    group_temp = group_temp.ewm(span = span).mean()
    fig = plt.figure(figsize = (10, 3))
    plt.plot(group_temp['date'], group_temp[column])
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title('%s VS Time(Date)' %column)

#Function to draw correlation:
def plot_correlation(df, columns_corr):
    colormap = plt.cm.RdBu
    plt.figure(figsize = (18, 15))
    sns.heatmap(news_rmv_outlier[columns_corr].astype('float').corr(),
        linewidths = 0.1, vmax = 1.0, vmin = -1.0, square = True,
        cmap = colormap, linecolor = 'white', annot = True)
    plt.title('Pair-wise Correlation')

#remove outliers with quantile:
def remove_outliers(df, column_list, low, high):
    temp_frame = df
    for column in column_list:



#Create new model from pre-trained models:
def create_model(input_shape, n_labels):
    input_tensor = Input(shape = input_shape)
    base_model = applications.VGG16(weights = None, include_top = False,
        input_tensor = input_tensor)
    base_model.load_weights('../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(2048, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_labels, activation = 'softmax', name = 'pre-trained-outputs')(x)
    model = Model(input_tensor, outputs)

    return model

######
def write_feature(MODEL, image_size):


#Integrated two models together from the pre-trained models:
from keras.layers import concatenate
def create_two_models(input_shape, n_labels):
    input_tensor = Input(shape = input_shape)

    #VGG16:
    base_model1 = applications.VGG16(weights = None, include_top = False,
        input_tensor = input_tensor)
    base_model1.load_weights('../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    #Inception Resnet V2:
    base_model2 = applications.InceptionResnetV2(weights = None,
        include_top = False, input_tensor = input_tensor)
    base_model2.load_weights('../input/keras-pretrained-models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')

    #Resnet 50:
    base_model3 = applications.ResNet50(weights = None, include_top = False, input_tensor = input_tensor)
    base_model3.load_weights('../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    ######
    #Combine the output:
    x1 = GlobalAveragePooling2D()(base_model1.output)
    x2 = GlobalAveragePooling2D()(base_model2.output)
    x3 = GlobalAveragePooling2D()(base_model3.output)
    x = concatenate([x1, x2, x3])

    x = BatchNormalization()(x)
    x = Dense(2048, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_labels, activation = 'softmax', name = 'pre-trained-outputs')(x)
    model = Model(input_tensor, outputs)

    return model

#Creating training_generator from the folder of images:
#We need to create a directory with all image files in it.
train_datagen = ImageDataGenerator(rescale = 1./255,
    validation_split = 0.2,
    horizontal_flip = False,
    vertical_flip = False)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = train,
    directory = '../input',
    x_col = 'id_code',
    y_col = 'diagnosis',
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    target_size = (HEIGHT, WIDTH),
    subset = 'validation')

########
#Weighted quadratic kappa score for keras:

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_pred = self.model.predict(X_val)

        _val_kappa = cohen_kappa_score(
            y_val.argmax(axis=1), 
            y_pred.argmax(axis=1), 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")

        return

#####
#TTA: test time augmentation: process to randomly applying some operations to the input data, by this mean,
#the model is never shown twice the exact same example and has to learn more general features about the classes he has to recognize.
###
#Perform random modifications to the test images, instead of showing the regular
#clean images, only once to the trained model, we will show it the augmented images several times.
#Then average the predictions of each corresponding image and take that as our final guess.
#In this model we perform 5 different types of augmentation on the test images and 
def TTA_keras(model, x_val):
    n = len(x_val)
    for i in range(len())


####
#resize images:
def process_image(image_path, desired_size = 224):
    im = Image.open(image_path)
    im = im.resize((desired_size, ) * 2, resample = Image.LANCZOS)
    return im 

def get_pad_width(im, new_shape, is_rgb = True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0] / 2), math.cell(pad_diff[0] / 2)
    l, r = math.floor(pad_diff[1] / 2), math.cell(pad_diff[1] / 2)
    if is_rgb:
        pad_width = ((t, b), (l, r), (0, 0))
    else:
        pad_width = ((t, b), (l, r))
    return pad_width


#######
import os
import pandas as pd
os.chdir('/Users/ruanhq/Desktop/Davis_PhD_Study/Jobs/Kaggle/House_regression')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#####
#To extract those observations that are outliers in 5 or more numerical features:
#input:
"""
: X is a list
: n_std means the threshold of outlier in which we want to delete.
"""
def uni_outlier_std(X, n_std):
    mean, std = X.mean(), X.std()
    cut_off = std * n_std
    lower_threshold = mean - cut_off
    higher_threshold = mean + cut_off
    return [True if xx < lower_threshold or xx > higher_threshold for xx in X]

def uni_outlier_iqr(X, k):
    ######
    q25, q75 = np.percentile(X, 25), np.percentile(X, 75)
    iqr = q75 - q25
    cut_off = iqr * k
    lower_threshold, higher_threshold = q25 - cut_off, q75 + cut_off

    return [True is xx < lower_threshold or xx > higher_threshold for xx in X]

#######
#Isolation forest for outlier detection:
import sklearn
from sklearn.ensemble import IsolationForest

#First 


#######
#Naive model averaging:
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   




########
#Function to train a variety of regression models:
def ensembling_classifier(X, X_test, y, model_type = 'lgb', n_split = 5, eval_metric = 'auc', columns = None, plot_feature_importance = False, verbose = 10000, early_stopping_rounds = 300, n_estimators = 9000, n_folds = 5, averaging = 'usual',
    second_order_model = 'SVM'):
    ######
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    ######
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
    'catboost_metric_name': 'AUC',
    'scoring_function': metrics.roc_auc_score}}
    ######
    oof = np.zeros((len(X), 1))
    prediction = np.zeros((len(X_test), 1))
    result_dict = {}

    #Create the out-of-fold predictions on train data:
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        ####
        X_train, X_valid = X[columns][train_index], X[columns][valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        ####
        if model_type = 'xgb':
            train_data = xgb.DMatrix(data = X_train, label = y_train, feature_names = X.columns)
            valid_data = xgb.DMatrix(data = X_valid, label = y_valid, feature_names = X.columns)
            ###
            Monitoring = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain = train_data, num_boost_round = n_estimators, evals = Monitoring, early_stopping_rounds = early_stopping_rounds)
            ##
            valid_pred_y = model.predict(xgb.DMatrix(X_valid, feature_names = X.columns), ntree_limit = model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names = X.columns), ntree_limit = model.best_ntree_limit)
            #####
        #Integrate more 
        if model_type = 'sklearn':
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_train, valid_pred_y))
            y_pred = model.predict_proba(X_test)
        #For catboost:
        if model_type = 'catboost':
            model = CatBoostClassifier(iterations = n_estimators,
                eval_metric = metrics_dict[eval_metric]['catboost_metric_name'], **params,
                loss_function = metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set = (X_valid, y_valid), cat_features = [], use_best_model = True, verbose = False)
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        #####
        oof_valid_index = y_pred_valid.reshape(-1, 1)
        scores.append(metrics_dict[eval_metric]['scoring_function'](y_train, valid_pred_y))
        #####
        prediction += y_pred.reshape(-1, 1)
    ####
    #Perform a second order inference:s
    if second_order_model = 'SVM':

    ####
    prediction /= n_splits
    print('CV_Mean_Score: {0:.5f}, std: {1:5f}.'.format(np.mean(scores), np.std(scores)))
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    return result_dict

#########

#Median stacking:
concat_sub['isFraud'] = concat_sub['ieee_median']
concat_sub[['TransactionID', 'isFraud']].to_csv('stack_median.csv',
    index = False, float_format = '%.6f')
######
#Average rank:



#########
#Calculate the AUROC:
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    n_false = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        n_false += (1 - y_i)
        auc += y_i * n_false
    auc /= (n_false *(n - n_false))
    return auc

