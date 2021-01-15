"""

AUX_FUNCTIONS_EXPLAIN.PY - Script imported in the notebooks with all the necessary functions for explaining the models

"""



def get_most_important_features(model, max_features = 20):
    
    '''
    function that returns a list with the most important features from an sklearn model object
    
    arguments
    -------------------------------------------------------
        model (sklearn model object) = already trained machine learning model from sklearn
        max_features (integer) = number of top features to extract. Default value is 20.
    
    '''
    import pandas as pd
    import numpy as np
    
    # get dictionary with importance scores
    feat_dict = model.get_booster().get_score(importance_type='weight')
    
    # create data frame 
    df = pd.DataFrame.from_dict(feat_dict, orient='index', columns=['importance']).reset_index()
    df.columns = ['features', 'importance']
    df.sort_values(by = 'importance', ascending = False, inplace = True)

    df = df[:max_features]

    features = [feat for feat in df['features']]
    
    return features


def plot_partial_dependence_visuals(model, X_test, max_features = 20, 
                                    title = 'Partial dependence of cancelation probability on most important features'):
    
    '''
    function that create a subplot of the partial dependence values for the most important features of a model up until max_features.
    
    arguments
    -------------------------------------------------------
        model (sklearn model object) = already trained machine learning model from sklearn
        X_test (pandas DataFrame) = Data Frame with features used for model training. Usually it is the testing data
        max_features (integer) = number of top features to extract. Default value is 20.
    
    '''
    
    import matplotlib.pyplot as plt 
    from sklearn.inspection import partial_dependence
    from sklearn.inspection import plot_partial_dependence
    from time import time

    print('Computing partial dependence plots...')
    tic = time()

    features = get_most_important_features(model = model, max_features = max_features)
    plot_partial_dependence(model, X_test, features, n_jobs=3, grid_resolution=20)

    print("done in {:.3f}s".format(time() - tic))
    fig = plt.gcf()
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5)
    
    return plt.show()


def plot_head_to_head_lift(new_pred, old_pred, X_test, y_test, bins = [-999, -0.1, -0.05, 0, 0.05, 0.1, 999], 
                               prem_var = "ERV_PRIMA_NETA_ANUALIZADA_PC"):
    

    '''
    function that plots a visualization to compare old models with new ones while also checking the actual values of a predictive modeling
    serves for validation of retention models
    
    arguments:
    ---------------------------------------------------------------------------
        new_pred (numpy array) = array with predicted probabilities from the new model
        old_pred (numpy array) = array with predicted probabilities from the  old model
        X_test (pandas DataFrame) = dataframe used for testing the model
        y_test (numpy array or Pandas Data Frame) = array with actual values of your target variable in the testing set
        bins (integer or list with integers) = number of bins used for the plots in the x-axis
        prem_var (string) = expiring premium variable name that should be in X_test
    '''
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    X_test = X_test.copy()

    X_test['old_preds'] = 1 - old_pred
    X_test['new_preds'] = new_pred
    X_test['pred_diff'] = X_test['new_preds'] - X_test['old_preds']
    X_test['actual'] = y_test
    X_test['premium'] = X_test[prem_var]


    X_test['pred_diff_bins'] = pd.cut(X_test['pred_diff'], bins = bins, right=False)

    data_plot = X_test.groupby(['pred_diff_bins']).agg({'premium': ['sum'],
                                                        'old_preds': ['mean'],
                                                        'new_preds': ['mean'],
                                                        'actual': ['mean']}).reset_index()
    
    data_plot.columns = ['Prediction_Difference','Total_Premium', 'Old_Predictions', 'New_Predictions', 'Observed']
    data_plot['Percent_Premium'] = data_plot['Total_Premium'] / sum(data_plot['Total_Premium'])

    # Start ploting
    sns.set_style(style="white")
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # First chart
    #ax1.bar(x =data_plot['Premium_Difference'].index, height = data_plot['Percent_Premium'], 
    #            color = 'gray', edgecolor = 'black', width = 0.2, label = "% Premium")
    sns.barplot(x = 'Prediction_Difference', y = 'Percent_Premium', data = data_plot, 
                color = 'gray',  edgecolor="black", label = 'Premium',
                ax = ax1)
    # formating
    ax1.set_xticks(np.arange(len(data_plot['Prediction_Difference'].index)))
    ax1.set_xticklabels(['< -10%', '-10%', '-5%', '0%', '5%', '>= 10%'], rotation = '0', fontsize = 15)
   
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.35, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')
    ax1.set_xlabel('Prediction Difference', fontsize = 20)
    ax1.set_ylabel('% Premium', fontsize = 20)


    # second chart
    ax2 = ax1.twinx()
    ax2.plot(data_plot['Old_Predictions'], linestyle='--', marker = '.', color = 'red', label = "Old Prediction")
    
    # formating
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.40, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')
    ax2.set_ylabel('Mean Predictions', fontsize = 20)
    
    # third chart
    ax2.plot(data_plot['New_Predictions'], linestyle='--', marker = '.', color = 'green', label = "New Predictions")
    ax2.set_yticks([])
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.45, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')

    # forth chart
    ax2.plot(data_plot['Observed'], linestyle='-', marker = '*', color = 'orange', label = "Observed")
    ax2.set_yticks([])
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.50, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')

    plt.title(f'Head-to-Head lift chart ', fontsize = 20)
    #print(data_plot)
    
    return plt.show()


def plot_old_vs_new_pred(new_pred, old_pred, X_test, y_test, bins, var):
    
    '''
    function that plots a visualization to compare old models with new ones while also checking the actual values of a predictive modeling grouped by a specified variable.
    Has the purpose of validating retention models
    
    arguments:
    ---------------------------------------------------------------------------
        new_pred (numpy array) = array with predicted probabilities from the new model
        old_pred (numpy array) = array with predicted probabilities from the  old model
        X_test (pandas DataFrame) = dataframe used for testing the model
        y_test (numpy array or Pandas Data Frame) = array with actual values of your target variable in the testing set
        bins (integer or list with integers) = number of bins used for the plots in the x-axis
        var (string) = variable name that is in analysis
    '''

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    X_test = X_test.copy()
    
    X_test['Bins'] = pd.cut(X_test[var], bins = bins)
    X_test['old_preds'] = 1 - old_pred
    X_test['new_preds'] = new_pred
    X_test['actual'] = y_test

    data_plot = X_test.groupby(['Bins']).agg({
                                           'old_preds': ['count', 'mean'],
                                           'new_preds': ['mean'],
                                           'actual': ['mean']}).reset_index()
    
    data_plot.columns = ['Bins', 'Nr_policies', 'Old_Predictions', 'New_Predictions', 'Observed']

    # Start ploting
    sns.set_style(style="white")
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # First chart
    sns.barplot(x = 'Bins', y = 'Nr_policies', data = data_plot, 
                color = 'gray',  edgecolor="black", label = 'Premium',
                ax = ax1)
    # formating
    ax1.set_xticks(np.arange(len(data_plot['Bins'].index)))
    ax1.set_xticklabels(data_plot['Bins'], rotation = '45', fontsize = 15)
   
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.35, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')
    ax1.set_xlabel(f'{var}', fontsize = 20)
    ax1.set_ylabel('Exposure', fontsize = 20)


    # second chart
    ax2 = ax1.twinx()
    ax2.plot(data_plot['Old_Predictions'], linestyle='--', marker = '.', color = 'red', label = "Old Prediction")
    
    # formating
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.40, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')
    ax2.set_ylabel('Mean Predictions', fontsize = 20)
    
    # third chart
    ax2.plot(data_plot['New_Predictions'], linestyle='--', marker = '.', color = 'green', label = "New Predictions")
    #ax2.set_yticks([])
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.45, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')

    # forth chart
    ax2.plot(data_plot['Observed'], linestyle='-', marker = '*', color = 'orange', label = "Observed")
    #ax2.set_yticks([])
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.50, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')

    plt.title(f'New vs Old Predictions by {var}', fontsize = 20)
    
    return plt.show()


def plot_old_vs_new_pred_categ(new_pred, old_pred, X_test, y_test, var):

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    X_test = X_test.copy()
    
    #X_test['Bins'] = pd.cut(X_test[var], bins = bins)
    X_test['old_preds'] = 1 - old_pred
    X_test['new_preds'] = new_pred
    X_test['actual'] = y_test

    data_plot = X_test.groupby([var]).agg({
                                           'old_preds': ['count', 'mean'],
                                           'new_preds': ['mean'],
                                           'actual': ['mean']}).reset_index()
    
    data_plot.columns = ['Bins', 'Nr_policies', 'Old_Predictions', 'New_Predictions', 'Observed']

    # Start ploting
    sns.set_style(style="white")
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # First chart
    sns.barplot(x = 'Bins', y = 'Nr_policies', data = data_plot, 
                color = 'gray',  edgecolor="black", label = 'Premium',
                ax = ax1)
    # formating
    ax1.set_xticks(np.arange(len(data_plot['Bins'].index)))
    ax1.set_xticklabels(data_plot['Bins'], rotation = '45', fontsize = 15)
   
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.35, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')
    ax1.set_xlabel(f'{var}', fontsize = 20)
    ax1.set_ylabel('Exposure', fontsize = 20)


    # second chart
    ax2 = ax1.twinx()
    ax2.plot(data_plot['Old_Predictions'], linestyle='--', marker = '.', color = 'red', label = "Old Prediction")
    
    # formating
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.40, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')
    ax2.set_ylabel('Mean Predictions', fontsize = 20)
    
    # third chart
    ax2.plot(data_plot['New_Predictions'], linestyle='--', marker = '.', color = 'green', label = "New Predictions")
    #ax2.set_yticks([])
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.45, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')

    # forth chart
    ax2.plot(data_plot['Observed'], linestyle='-', marker = '*', color = 'orange', label = "Observed")
    #ax2.set_yticks([])
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.50, 0.5, 0.5), fontsize = 'x-large', edgecolor = 'white')

    plt.title(f'New vs Old Predictions by {var}', fontsize = 20)
    
    return plt.show()


def load_trained_models(file, bucket_name, path):
    '''
    function that loads an already trained model into the notebook.
    
    arguments:
    --------------------------------------------------------------------------
        file (string) = file to be loaded. Example: "mymodel.sav"
        bucket_name (string) = name of the amazon s3 bucket where to dump/read data.
        path (string) = path to save the trained model. Example: "/My/Model/is/here/"
        option (string) = choose only 'save' for saving models or 'load' for loading models
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
        
    return model


def get_auc_on_test(y, pred, label):
    
    '''
    function that calculates auc of a set of predictions.
    
    arguments:
    --------------------------------------------------------------------------

    '''   
    import sklearn.metrics
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=label)
    auc_test = metrics.auc(fpr, tpr)
    print(f"Auc on test = {round(auc_test,2)}")
    return auc_test
