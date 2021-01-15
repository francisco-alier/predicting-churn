
def bootstrap_auc(models, X_test, y_test, targets, bootstraps = 100, fold_size = 1000):
    '''
    function that returns an [m * n] matrix with bootstrap results. m represents the number of target variables and and n the number of     bootstrap samples. 
    
    arguments:
        models (dictionary) = dictionary with model objects and labels. Example: {'models':[model1], 'labels':[label1]}
        X_test (pandas DataFrame) = dataframe used for testing the model
        y_test (numpy array or Pandas Data Frame) = array with actual values of your target variable in the testing set
        targets (list) = list with strings of the target variables
        bootstraps (integer) = number of bootstrap samples to perform
        fold_size (integer) = number of  
    '''    
    
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    
    statistics = np.zeros((len(targets), bootstraps))
    
    for model in models['classifiers']:
        
        pred = model.predict_proba(X_test)[:,1]
        
        for c in range(len(targets)):
            df = pd.DataFrame(columns=['y', 'pred'])
            df.loc[:, 'y'] = y_test
            df.loc[:, 'pred'] = pred
            # get positive examples for stratified sampling
            df_pos = df[df.y == 1]
            df_neg = df[df.y == 0]
            prevalence = len(df_pos) / len(df)
            
            for i in range(bootstraps):
                # stratified sampling of positive and negative examples
                pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
                neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

                y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
                pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
                score = roc_auc_score(y_sample, pred_sample)
                statistics[c][i] = score
    
    return statistics