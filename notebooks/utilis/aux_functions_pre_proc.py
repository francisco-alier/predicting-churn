"""

AUX_FUNCTIONS.PY - Script imported in the notebooks with all the necessary pre-processing functions

It is divided in the following categories:
    - pre-precessing
    - visualizations
    - feature engineering
"""

###########################################################################################
### ################################## PRE-PROCESSING FUNCTIONS ###########################
###########################################################################################
def check_missings(df):
    
    """
    function to print out missing values for each variable in a data frame
    ----------------------------------------------------------------------------------------
    arguments:
     - df: data frame to be analyzed
    """
    
    import numpy as np
    import pandas as pd
    
    #make a list of the variables that contain missing values
    vars_with_na = [var for var in df.columns if df[var].isnull().sum()>1]
    miss_pred = pd.isnull(df[vars_with_na]).sum().sort_values(ascending=False)
    # print the variable name and the percentage of missing values
    if not vars_with_na:
        print("There are no missing values")

    else:
        for var in miss_pred.index:
            missing = np.round(df[var].isnull().mean(), 3)
            print(f"{var} : {missing:.3%} missing values")
            
            
def remove_outliers(df, var):
    """
    function to remove outlier data from a defined variable based on the IQR
    ----------------------------------------------------------------------------------------
    arguments:
     - df (pandas Data Frame): data frame to be analyzed
     - var (string): string of the variable to be analyzed
     
    """
    import numpy as np
    
    df = df.copy()
    
   # remove outliers
    Q1 = np.nanquantile(df[var] ,0.25)
    Q3 = np.nanquantile(df[var], 0.75)
    IQR = Q3 - Q1
    
    lower_end = Q1 - 1.5 * IQR 
    high_end = Q3 + 1.5 * IQR 
    
    df_filtered = df.drop(df[(df[var] < lower_end) | (df[var] > high_end)].index)
    
    return df_filtered

###########################################################################################
### ################################## VISUALIZATION FUNCTIONS ###########################
###########################################################################################

def visualize_outliers(df, var):
    """
    function to visualize outliers by looking at % of observations between a categorical variable
    ----------------------------------------------------------------------------------------
    arguments:
     - df (pandas Data Frame): data frame to be analyzed
     - var (string): string of the variable to be analyzed
     
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    num_var = df.groupby(var)[var].count() 
    total = np.float(len(df))
    
    var_perc = num_var / total 
    
    var_perc.plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    
    return plt.show()


def visualize_tgt_by_categorical(df, var, target):
    """
    function to visualize bar plots by categorical value of average values of the target
    ----------------------------------------------------------------------------------------
    arguments:
         df (pandas Data Frame): data frame to be analyzed
         var (string) = variable to be analyzed on the x axis
         target (string) = string of target variable that is being analyzed. Needs to have 0 or 1 levels
     
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.figure(figsize=(10,5))
    
    grouped_values = df.groupby(var)[target].mean().sort_values(ascending = False).reset_index()

    sns.set(style = 'white')
    sns.barplot(x = var, y = target, data = grouped_values, palette = sns.color_palette("RdBu", n_colors = 7))

    return plt.show()
    
    
def tgt_dist(df, y_var, target):
    '''
    creates distribution plots for classifcation targets based on a single variable
    ----------------------------------------------------------------------------------------
    arguments:
        df (pandas data frame) = data frame of analysis
        var (string) = variable to be analyzed on the x axis
        target (string) = string of target variable that is being analyzed. Needs to have 0 or 1 levels
    
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    df = df.copy()
    df = remove_outliers(df = df, var = y_var)
    
    # check if tarhet is from classification problem
    if len(df[target].unique()) <= 2:
        
        target_0 = df.loc[df[target] == 0]
        target_1 = df.loc[df[target] == 1]

        sns.set(style = 'white')
        plot_0 = sns.distplot(target_0[[y_var]], hist=False, kde = True, label = '0')
        plot_1 = sns.distplot(target_1[[y_var]], hist=False, kde = True, label = '1')

        plt.xlabel(y_var)

        plt.legend()
        plt.show()

    else:
        print(f'Target {target} has more than 2 categories!')
        

def tgt_dist_violin(df, x_var, y_var, target):
    '''
    creates violin plots for classifcation targets based on a single variable
    ----------------------------------------------------------------------------------------
    arguments:
        df (pandas data frame) = data frame of analysis
        x_var (string) = variable to be analyzed on the x axis
        y_var (string) = variable to be analyzed on the y axis
        target (string) = string of target variable that is being analyzed. Needs to have 0 or 1 levels
    
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # check if tarhet is from classification problem
    if len(df[target].unique()) <= 2:
        
        # remove outliers
        df = df.copy()
        df = remove_outliers(df = df, var = y_var)
    
        # Plot
        sns.set(style = 'white')
    
        plot1 = sns.violinplot(y = y_var, hue = target, data = df, split = True, palette = sns.color_palette("RdBu", n_colors=7))
        plot1.set_ylabel(y_var)

        plt.show()

    else:
        print(f'Target {target} has more than 2 categories!')


def tgt_dist_barplots(df, x_var, y_var, target):
    '''
    creates violin plots for classifcation targets based on two variables
    ----------------------------------------------------------------------------------------
    arguments:
        df (pandas data frame) = data frame of analysis
        x_var (string) = variable to be analyzed on the x axis
        y_var (string) = variable to be analyzed on the y axis
        target (string) = string of target variable that is being analyzed. Needs to have 0 or 1 levels
    
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # check if tarhet is from classification problem
    if len(df[target].unique()) <= 2:
        
        # remove outliers
        df = df.copy()
        df = remove_outliers(df = df, var = y_var)
    
        # Plot
        sns.set(style = 'white')
        plot = sns.barplot(y = y_var, hue = target, data = df, palette = sns.color_palette("RdBu", n_colors=7))

        plot.set_ylabel(y_var)

        return plot
        
    else:
        print(f'Target {target} has more than 2 categories!')
        
        
def tgt_dist_countplots(df, x_var, target):
    '''
    creates bar plots for classifcation targets based on a categorical variable
   ----------------------------------------------------------------------------------------
    arguments:
        df (pandas data frame) = data frame of analysis
        x_var (string) = variable to be analyzed on the x axis
        target (string) = string of target variable that is being analyzed. Needs to have 0 or 1 levels
    
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # check if tarhet is from classification problem
    if len(df[target].unique()) <= 2:
    
        # Plot
        sns.set(style = 'white')
        plot = sns.countplot(x = x_var, hue = target, data = df, palette = sns.color_palette("Set1", n_colors=8, desat=.5))

        plot.set_xlabel(x_var)

        return plot
        
    else:
        print(f'Target {target} has more than 2 categories!')
        
        
def plot_correlations(df, var):
    '''
    creates a correlation plot for a user specified variable comparing with all the variables in the data frame
    ----------------------------------------------------------------------------------------
    arguments:
        df (pandas data frame) = data frame of analysis
        var (string) = variable to be analyzed on the x axis
    '''
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(f"Correlation analysis of {var}", fontsize=12)
    
    corr = df.corr()
    for id_chart, i in zip(range(1,9), range(0, len(corr), 10)):
    
        corr1 = corr.iloc[i:i+10,:]

        plt.subplot(2,4,id_chart)
        plot = sns.heatmap(corr1[[var]].sort_values(by=[var],ascending=False), 
                           annot = True, 
                           cbar_kws= {'orientation': 'horizontal'},
                           cmap="Blues")
        
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=6)
        
        plt.subplots_adjust(wspace=2)
    
    return plot


def plot_unique_figure_correlations(df, var, type = 'pos'):
    
    '''
    creates a correlation plot for a user specified variable comparing the top 5 most correlated variables
    ----------------------------------------------------------------------------------------
    arguments:
        data (pandas data frame) = data frame of analysis
        var (string) = variable to be analyzed on the x axis
        type (string) = string with options "neg" or "pos" to select top 5 with most positive correlation or more negative correlation
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig = plt.figure(figsize=(5,10))
    fig.suptitle(f"Correlation analysis of {var}", fontsize=12)
    
    corr = df.corr()
    
    if type == 'pos':
        plot = sns.heatmap(corr[[var]].sort_values(by=[var],ascending=False)[1:6], 
                           annot = True, 
                           cbar_kws= {'orientation': 'vertical'},
                           cmap="Blues",
                           annot_kws= {"size":12})
        
        top_5 = list(corr[[var]].sort_values(by=[var],ascending=False)[1:6].index)
        print(f"the top 5 most correlated features with {var} are {top_5}")

    else:
        plot = sns.heatmap(corr[[var]].sort_values(by=[var],ascending=False)[-5:], 
                           annot = True, 
                           cbar_kws= {'orientation': 'vertical'},
                           cmap="Blues",
                           annot_kws= {"size":12})
        
        less_5 = list(corr[[var]].sort_values(by=[var],ascending=False)[-5:].index)
        print(f"the top 5 most negative correlated features with {var} are {less_5}")
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.subplots_adjust(wspace=2)
    
    #get top 5 in list
    
    
    
    return plot

###########################################################################################
### ################################## FEATURE ENGINEERING FUNCTIONS ######################
###########################################################################################


def get_constant_features(df):
    '''
    returns a list with features that are constant in the data frame, i.e, the ones without variability
    ----------------------------------------------------------------------------------------
    arguments:
        df (pandas data frame) = data frame of analysis
    '''
    import pandas as pd
    
    constant_features = [feat 
                         for feat in df.columns 
                         if df[feat].dtype != 'O' and df[feat].std() == 0]
    
    return constant_features


def get_duplicated_features(df):
    '''
    returns a list with features that are duplicated
    arguments:
    df (pandas data frame) = data frame of analysis
    '''
    
    
    duplicated_feat = dict()
    for i in range(0, len(df.columns)):
    #if i % 10 == 0: # this helps me understand how the loop is going
    # print(i)

        col_1 = df.columns[i]

        for col_2 in df.columns[i + 1:]:
            if df[col_1].equals(df[col_2]):
                duplicated_feat.update({col_2:f"{col_1} is equal to {col_2}"})

    return duplicated_feat;


def intersection(lst1, lst2):
    '''
    function that returns a list which match between lst1 and lst2
    ----------------------------------------------------------------------------------------
    arguments:
        lst1 (list) = list number 1
        lst2 (list) = list number 2
    '''
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3


def correlation(df, threshold):
    '''
    function that returns a set with any feature in the data frame that has a correlation with some other feature above a threshold
    
    arguments:
        df (pandas DataFrame) = dataframe to be analyzed - only select for usage with numerical, integers or floting point variables
        threshold (floating point) = value to filter the correlation values 
                                    (example - a value of 0.8 would select the features with a correlation above 80%)
    '''
    col_corr = dict()  # Set of all the names of correlated columns
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.update({colname:f"corr with {corr_matrix.columns[j]}"})
    return col_corr



def get_individual_roc_values(X_train, X_test, y_train, y_test):
    '''
    Function that returns a pandas Series with individual roc calculations for a training dataset.
    
    Behind the scenes builds a Decision tree classifier for each of the different features fiting it to the training set and calculates the roc values 
    ----------------------------------------------------------------------------------------
    arguments:
        X_train (pandas DataFrame) = dataframe used for training the model
        X_test (pandas DataFrame) = dataframe used for testing the model
        y_train (numpy array or Pandas Data Frame) = array with actual values of your target variable in the training set
        y_test (numpy array or Pandas Data Frame) = array with actual values of your target variable in the testing set
        
    '''
    
    roc_values = []
    for feature in X_train.columns:
        clf = DecisionTreeClassifier()
        clf.fit(X_train[feature].to_frame(), y_train)
        
        y_scored = clf.predict_proba(X_test[feature].to_frame())
        roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
    
    return pd.Series(roc_values)