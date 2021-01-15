"""

AUX_FUNCTIONS_SMOOTHED.PY - Script imported in the notebooks with all the necessary functions for smoothing ML models

"""
###
def log_odds(p):
    '''
    calculates the log odds of a single value of p (probability)
    
    arguments
    -------------------------------------------------------
        p (float) = floating point number with domain of 0 to 1. Usually a probability. 
    '''
    import numpy as np
    
    odds = (p / (1-p))
    log_odds = np.log(odds)
    
    return log_odds


def calculate_linear_coefficients(rate_changes, log_odds, weights):
    
    '''
    function that returns a pandas Data Frame with the coefficients a and b of fitted linear regression to the log-odds of ML predictions 
    
    arguments
    -------------------------------------------------------
        rate_changes (numpy array) = numpy array with different rate changes to test
        log_odds (pandas DataFrame) = Data Frame with calculated log odds. It should be on the form where the columns are the log odds for each rate change presented in the array rate_changes
        weights (numpy array) = for calculating weighted predictions. It should have the same shape as rate_changes.
    
    '''
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import time
    
    start = time.time()
    
    coeff_linear = pd.DataFrame(index = range(0, len(log_odds)), columns = {'a_linear', 'b_linear'}).fillna(0)
    
    for i in range(0, len(coeff_linear), 1):
        lm = LinearRegression()
        lm.fit(X = rate_changes.reshape(-1,1), 
               y = log_odds.iloc[i,:].values.reshape(-1, 1),
               sample_weight = np.array(weights))

        #Get coefficients and append
        coeff_linear.loc[i, 'a_linear'] = lm.intercept_
        coeff_linear.loc[i, 'b_linear'] = lm.coef_[0]
            
    end = time.time()
    print(f'the program took {end - start} seconds to run')
    
    return coeff_linear


def calculate_quadratic_coefficients(rate_changes, log_odds, weights):
    
    '''
    function that returns a pandas Data Frame with the coefficients a and b of fitted linear regression to the log-odds of ML predictions 
    
    arguments
    -------------------------------------------------------
        rate_changes (numpy array) = numpy array with different rate changes to test
        log_odds (pandas DataFrame) = Data Frame with calculated log odds. It should be on the form where the columns are the log odds for each rate change presented in the array rate_changes
        weights (numpy array) = for calculating weighted predictions. It should have the same shape as rate_changes.
    
    '''
    
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import time
    
    start = time.time()
    
    coeff_quad = pd.DataFrame(index = range(0, len(log_odds)), columns = {'a_quadratic', 'b_quadratic', 'c_quadratic'}).fillna(0)
    
    for i in range(0, len(coeff_quad), 1):
    
        orig = np.array(rate_changes).reshape(-1,1)
        X_2 = np.c_[orig, orig**2]
        lm2 = LinearRegression()

        lm2.fit(X = X_2,
                y = log_odds.iloc[i,:].values.reshape(-1,1),
                sample_weight = np.array(weights)
                )

        coeff_quad.loc[i, 'a_quadratic'] = lm2.intercept_[0]
        coeff_quad.loc[i, 'b_quadratic'] = lm2.coef_[0,0]
        coeff_quad.loc[i, 'c_quadratic'] = lm2.coef_[0,1]
        
    end = time.time()
    print(f'the program took {end - start} seconds to run')
    
    return coeff_quad


def calculate_cubic_coefficients(rate_changes, log_odds, weights):    
    '''
    function that returns a pandas Data Frame with the coefficients a, b, c and d of fitted linear regression with a 3 degree polynomial fit to the log-odds of ML predictions 
    
    arguments
    -------------------------------------------------------
        rate_changes (numpy array) = numpy array with different rate changes to test
        log_odds (pandas DataFrame) = Data Frame with calculated log odds. It should be on the form where the columns are the log odds for each rate change presented in the array rate_changes
        weights (numpy array) = for calculating weighted predictions. It should have the same shape as rate_changes.
    
    '''
    
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import time
    
    start = time.time()
    
    coeff_cubic = pd.DataFrame(index = range(0, len(log_odds)), columns = {'a_cubic', 'b_cubic', 'c_cubic', 'd_cubic'}).fillna(0)
    
    orig = np.array(rate_changes).reshape(-1,1)
    X_3 = np.c_[orig, orig**2, orig**3]

    # Fitting Polynomial Regression to the dataset
    for i in range(0, len(coeff_cubic), 1):

        lm3 = LinearRegression()
        lm3.fit(X = X_3, 
                y = log_odds.iloc[i,:].values.reshape(-1, 1),
                sample_weight = np.array(weights)
                )

        #Get coefficients and append
        coeff_cubic.loc[i, 'a_cubic'] = lm3.intercept_[0]
        coeff_cubic.loc[i, 'b_cubic'] = lm3.coef_[0,0]
        coeff_cubic.loc[i, 'c_cubic'] = lm3.coef_[0,1]
        coeff_cubic.loc[i, 'd_cubic'] = lm3.coef_[0,2]
        
    end = time.time()
    print(f'the program took {end - start} seconds to run')
    
    return coeff_cubic


def calculate_smooth_linear(a, b, x, c = 0, d = 0):
    '''
    calculates smoothed probability based on a simple linear regression
    
    '''
    import numpy as np
    
    log_odds = a + (b * x)
    smooth_pred = 1 / (1 + np.exp((-log_odds)))
    
    return smooth_pred


def calculate_smooth_quadratic(a, b, x, c, d = 0):
    '''
    calculates smoothed probability based on a second order linear regression
    
    '''
    import numpy as np
    
    log_odds = a + b * x + c * (x**2)
    smooth_pred = 1 / (1 + np.exp(-(log_odds)))
    
    return smooth_pred


def calculate_smooth_cubic(a, b, x, c, d):
    '''
    calculates smoothed probability based on a third order linear regression
    '''
    import numpy as np
    
    log_odds = a + b * x + c * (x**2) + d * (x**3)
    smooth_pred = 1 / (1 + np.exp(-(log_odds)))
    
    return smooth_pred

                 
def calculate_smooth_predictions(a, b, x, c = 0, d = 0, degree = "linear"):
    '''
    calculates smoothed probability based on a simple linear regression
    
    arguments
    -------------------------------------------------------
        a (numpy array) = array with intercepts from fitted linear curve
        b (numpy array) = array with first order degree coefficients from fitted linear curve
        c (numpy array) = array with second order degree coefficients from fitted linear curve. If not specified defaults to zero.
        d (numpy array) = array with third order degree coefficients from fitted linear curve. If not specified defaults to zero.
        x (numpy array) = array with different rate changes or known predictors
        degree (str) = one of "linear", "quadratic" or "cubic". Specifies the smoothed predictions calculations. 
    '''
    # Master
    smooth_calculations = {
                            "linear": calculate_smooth_linear,
                            "quadratic": calculate_smooth_quadratic,
                            "cubic": calculate_smooth_cubic
                          }
    
    if degree in smooth_calculations:
        
        selected_function = smooth_calculations[degree]
        
        smooth_pred = selected_function(a, b, x, c, d)

    else:
        print(f'Unknown degree of {degree}. Please select one of linear/quadratic/cubic.')
        
    return smooth_pred
