"""

AUX_FUNCTIONS_TRAIN.PY - Script imported in the notebooks with all the necessary functions for modeling

"""

###########################################################################################
### ################################## TESTING MODELS FUNCTIONS ###########################
###########################################################################################

def print_roc_auc_scores(models, X, y):
    '''
    function that prints out the roc-auc score for a given model
    -----------------------------------------------------------------------
    arguments:
        models (dictionary) = dictionary with model objects and labels. Example: {'classifiers':[classifier1], 'labels':[label1]}
        X (pandas DataFrame) = dataframe with the features used for the model
        y (numpy array or Pandas Data Frame) = array with actual values of your target variable in the training set
    '''
    from sklearn.metrics import roc_auc_score

    for classifier, label in zip(models['classifiers'], models['labels']):
        
        preds = classifier.predict_proba(X)[:,1]

        print(f'{label}:')
        print(f' - Train set roc-auc: {roc_auc_score(y, preds):.5%}')
        #print(f' - Test set roc-auc: {roc_auc_score(y_test, X_test_preds):.5%}')
        print('\n=========================\n')


def print_cv_mean_roc_auc_scores(models, X, y, k):
    '''
    function that prints out the mean and std of roc-auc score for a given model using k-fold cross validation
    -----------------------------------------------------------------------
    arguments:
        models (dictionary) = dictionary with model objects and labels. Example: {'classifiers':[classifier1], 'labels':[label1]}
        X (pandas DataFrame) = dataframe with the features used for the model
        y (numpy array or Pandas Data Frame) = array with actual values of your target variable in the training set
        k (integer) = number of folds for cross-validation scores
    '''
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    for classifier, label in zip(models['classifiers'], models['labels']):
        
        accuracies = cross_val_score(estimator = classifier, X = X, y = y.values.ravel(), cv = k, scoring='roc_auc')

        CrossValMean = accuracies.mean()
        CrossValSTD = accuracies.std()
        
        print(f'{label}:')
        print(f' - Final CrossValMean:: {CrossValMean:.5%}')
        print(f' - Final CrossValSTD: : {CrossValSTD:.5%}')
        print('\n=========================\n')
    
# Generate ROC curve values: fpr, tpr, thresholds
def plot_roc_curves(models, X_test, y_test, model_type = 'h2o'):
    '''
    function that prints out the roc-auc score for the training set and the test set of a given model
    -----------------------------------------------------------------------
    arguments:
        models (dictionary) = dictionary with model objects and labels. Example: {'classifiers':[model1], 'labels':[label1]}
        X_test (pandas DataFrame) = dataframe used for testing the model
        y_test (numpy array or Pandas Data Frame) = array with actual values of your target variable in the testing set
        model_type (string) = Option with 'h2o' or 'sklearn'
    '''
    from sklearn.metrics import roc_auc_score, roc_curve
    import h2o
    import matplotlib.pyplot as plt
    import pandas as pd
    import warnings
    
    warnings.filterwarnings("ignore")

    result_table = pd.DataFrame(columns=['labels', 'fpr','tpr','auc'])
    
    for model, label, color in zip(models['classifiers'], models['labels'], models['colors']):
        
        if model_type == 'h2o':
            # calculate predictions
            y_preds  = model.predict(X_test)[:,2]
            y_preds = h2o.as_list(y_preds, use_pandas=True)
        else:
            y_preds  = model.predict_proba(X_test)[:,1]
        
        # extract false positive rates and true positive rates
        fpr, tpr, thresholds = roc_curve(y_test, y_preds)
        auc = roc_auc_score(y_test, y_preds)
        
        result_table = result_table.append({'labels':label,
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc,
                                            'color': color}, ignore_index=True)
        
   
    result_table.set_index('labels', inplace=True)
    
    # create figure and axe
    fig, ax = plt.subplots()
    
    for i in result_table.index:

        ax1 = plt.plot(result_table.loc[i]['fpr'], 
                       result_table.loc[i]['tpr'], 
                       label=f"{i}: auc= {result_table.loc[i]['auc']:.2%}",
                       color = result_table.loc[i]['color'])
        
    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    leg = ax.legend()
    
    return ax1


def plot_calibration_curve(models, X_test, y_test, y_old, n_bins = 20, model_type = 'h2o', plat_scalling = False):
    
    '''
    function that returns plots of calibrated probabilities of a classification model. 
    -----------------------------------------------------------------------
    arguments:
        models (dictionary) = dictionary with model objects and labels. Example: {'models':[model1], 'labels':[label1]}
        X_test (pandas DataFrame) = dataframe used for testing the model
        y_test (numpy array or Pandas Data Frame) = array with actual values of your target variable in the testing set
        y_old (numpy array or Pandas Data Frame) = array with old model predictions, in this case from GLM models
        n_bins (integer) = number of bins used for the plots
        model_type (string) = Option with 'h2o' or 'sklearn'
        plat_scalling (boolean) = True or False depending if the calibration uses the plat scalling method
    '''
    import h2o
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LogisticRegression as LR 
    
    warnings.filterwarnings("ignore")
    
    result_table = pd.DataFrame(columns=['labels', 'fraction_pos','mean_pred'])
    
    print_old = True
    
    for model, label, color in zip(models['classifiers'], models['labels'], models['colors']):
        
        if model_type == 'h2o':
            # calculate predictions
            pred  = model.predict(X_test)[:,2]
            pred = h2o.as_list(pred, use_pandas=True).values
            pred_calibrated = np.zeros_like(pred)
            
            # For plat scalling
            lr = LR(solver='liblinear', max_iter=10000)
            lr.fit(pred.reshape(-1, 1), y_test)    
            pred_calibrated = lr.predict_proba(pred.reshape(-1, 1))[:,1]
            
        else:
            pred = model.predict_proba(X_test)[:,1]
            pred_calibrated = np.zeros_like(pred)
            
            # For plat scalling
            lr = LR(solver='liblinear', max_iter=10000)
            lr.fit(pred.reshape(-1, 1), y_test)    
            pred_calibrated = lr.predict_proba(pred.reshape(-1, 1))[:,1]
            
        # extract fraction of positives and check for plat scalling argument
        if plat_scalling == True:
            pred = pred_calibrated
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, pred, n_bins=n_bins)
        
        result_table = result_table.append({'labels': label,
                                            'preds': pred_calibrated,
                                            'fraction_pos': fraction_of_positives, 
                                            'mean_pred': mean_predicted_value,
                                            'color': color
                                            }, ignore_index=True)
        
        if print_old == True:
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_old, n_bins=n_bins)
            result_table = result_table.append({'labels': 'GLM',
                                                'preds': y_old,
                                                'fraction_pos': fraction_of_positives, 
                                                'mean_pred': mean_predicted_value,
                                                'color': '#138D75'
                                                }, ignore_index=True)
            
            print_old = False
        
    # create figure and axe
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], linestyle='--', color = 'black', label = 'Perfectly Calibrated Plot')
    
    for i in result_table.index:
        
        plot1 = ax1.plot(result_table.loc[i]['mean_pred'], 
                         result_table.loc[i]['fraction_pos'],
                         label = result_table.loc[i]['labels'],
                         marker='.',
                         color = result_table.loc[i]['color'])
        
        plot2 = sns.distplot(result_table.loc[i]['preds'], 
                             hist = False, 
                             bins = n_bins, 
                             label = result_table.loc[i]['labels'],
                             color = result_table.loc[i]['color'],
                             norm_hist = False,
                             ax = ax2
                            )
        
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper left")
    ax1.set_title('Calibration plots')

    ax2.set_xlabel("Mean predicted value")
    #ax2.set_ylim([0, len(X_test)])
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper right", ncol=2)
    
    return plot1, plot2


def print_classification_reports(models, X_test, y_test):
    
    '''
    function that prints out the confusion matrixes for the training set and the test set of a list of models
    -----------------------------------------------------------------------
    arguments:
        models (dictionary) = dictionary with model objects and labels. Example: {'models':[model1], 'labels':[label1]}
        X_test (pandas DataFrame) = dataframe used for testing the model
        y_test (numpy array or Pandas Data Frame) = array with actual values of your target variable in the testing set
    '''
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    
    for model, label in zip(models['classifiers'], models['labels']):
    
        pred = model.predict(X_test)
        
        report = classification_report(y_test, pred)
        matrix = confusion_matrix(y_test, pred)
        
        matrix_df = pd.DataFrame(matrix)
        matrix_df.columns = ['Predicted 0', 'Predicted 1']
        matrix_df.index = ['Actual 0', 'Actual 1']
        
        print(f"Model analyzed: {label} \n")
        print(matrix_df)
        print(report)
        print("==============================")
        
        
def plot_actual_pred_by_var(X_test, var, pred, y_test, bins):
    '''
    function that plots a visualization to check average predicted vs. actual values by an input variable
    -----------------------------------------------------------------------
    arguments:
        X_test (pandas DataFrame) = dataframe used for testing the model
        var (string) = variable to analyze on the x-axis
        pred (numpy array) = array with predicted probabilities from the model
        y_test (numpy array or Pandas Data Frame) = array with actual values of your target variable in the testing set
        bins (integer) = number of bins used for the plots in the x-axis
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    X_test['Bins'] = pd.cut(X_test[var], bins = bins)
    X_test['pred'] = pred
    X_test['y'] = y_test
    
    data_plot = X_test.groupby(['Bins']).agg({'pred': ['count', 'mean'],
                                     'y': ['mean']}).reset_index()

    data_plot.columns = ['Bins', 'Nr_Policies', 'Avg_pred', 'Actual']
    
    # Start ploting
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # First chart
    ax1.bar(x =data_plot['Bins'].index, height = data_plot['Nr_Policies'], 
            color = 'gray', edgecolor = 'black', width = 1, label = "Exposure")
    
    # formating
    ax1.set_xticks(np.arange(len(data_plot['Bins'].index)))
    ax1.set_xticklabels(data_plot['Bins'], rotation = '45', fontsize = 15)
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.40, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')
    ax1.set_xlabel(f'{var}', fontsize = 20)
    ax1.set_ylabel('Exposure', fontsize = 20)


    # second chart
    ax2 = ax1.twinx()
    ax2.plot(data_plot['Avg_pred'], linestyle='--', marker = '.', color = '#0039FF', label = "Predicted Values")
    
    # formating
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.45, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')
    ax2.set_yticks(np.arange(0, 0.5, 0.02))

    # third chart
    ax3 = ax1.twinx()
    ax3.plot(data_plot['Actual'], linestyle='-', marker = '*', color = '#020B28', label = "Actual Values")
    ax3.set_yticks([])

    ax3.legend(loc='upper left', bbox_to_anchor=(0, 0.5, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')

    plt.title(f'Actual vs. Predictions by {var}', fontsize = 20)
    
    return plt.show()

###########################################################################################
### ################################## DUMP/LOAD FUNCTIONS ###########################
###########################################################################################

def dump_trained_models(models, bucket_name, path):
    
    '''
    function that saves an already trained model objects in s3 buckets or loads them into the notebook. 
    If loading, it will create model objects that are defined in the dictionary models['classifiers'].
    -----------------------------------------------------------------------
    arguments:
        models (dictionary) = dictionary with model objects and filenames. Example: {'models':[model1], 'filenames':['model.sav']}
        bucket_name (string) = name of the amazon s3 bucket where to dump/read data.
        path (string) = path to save the trained model. Example: "/My/Model/is/here/"
    '''    
    import tempfile
    import boto3
    import joblib
    import pickle
    
    s3 = boto3.resource('s3')

    for model, file in zip(models['classifiers'], models['filenames']):

        path = path
        model_filename = file

        OutputFile = path + model_filename

        # WRITE
        pickle_byte_obj = pickle.dumps(model)
        s3.Object(bucket_name, OutputFile).put(Body=pickle_byte_obj)

        print(f"{file} successfully saved!")
            

def load_trained_models(file, bucket_name, path):
    '''
    function that loads an already trained model into the notebook.
    -----------------------------------------------------------------------
    arguments:
        file (string) = file to be loaded. Example: "mymodel.sav"
        bucket_name (string) = name of the amazon s3 bucket where to dump/read data.
        path (string) = path to save the trained model. Example: "/My/Model/is/here/"
    '''    
    import tempfile
    import boto3
    import joblib
    import pickle
    
    s3 = boto3.resource('s3')
    
    path = path
    model_filename = file

    OutputFile = path + model_filename

    my_dictionary_bytes = s3.Object(bucket_name, OutputFile).get()['Body'].read()

    model = pickle.loads(my_dictionary_bytes)
    
    print(f'Model {model_filename} was successfully loaded!')
        
    return model