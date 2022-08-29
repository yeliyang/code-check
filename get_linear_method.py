import numpy as np
try:
    from sklearn.linear_model import LinearRegression
except:
    import os
    os.system('pip install sklearn')


def validlist(array, lens = 2):
    """Check list validation.
    
    Retrieves (timestamp, value) list and check if list has length 
    longer than limit, default is 2.
    
    Args:
        array: A list contains tuple(timestamp, value)
        lens: Minimum length that array should have
        
    Returns:
        The samllest index that forms a valid array. If no valid index found,
        return -1.
        
    """
    if len(array) < lens:
        # array has to greater than limit range
        return 0
    return 1


def get_linear_intercept(data : list, verbose = 1):
    """Get linear intercept to predict future value
    
    Assume data obey linear method, we use linear model to fit dataset
    and get the coef and intercept to predict future value.
    
    Args:
        data:  A list contains tuple(timestamp, value)
        verbose: [0, 1] option, default is 1.
                 0 to not plot figure.
                 1 to plot figure.
    Returns:
        Tuple(coef, intercept)
        example:
        
        (47.657499999999985, 816.6703571428571)
        
        A tuple contains the coef and intercept that can forms a linear method
        y = x * coef + intercept
    """
    index = validlist(data, lens=2)
    # check the validation of the input data list,
    # if return is 0, means list is invalid, return (0,0)
    if not index:
        return (0, 0)
    # transfor list into numpy array and seperate into x and y
    data = np.array(data)
    train_x = data[:, 0].reshape(-1, 1)
    train_y = data[:, 1]
    reg = LinearRegression().fit(train_x, train_y)
    pred_y = train_x * reg.coef_[0] + reg.intercept_
    if verbose:
        plotting(train_x, train_y, pred_y)
    return (reg.coef_[0], reg.intercept_)


def plotting(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.plot(x, y, label = 'true')
    plt.plot(x, pred_y, label = 'pred')
    plt.legend()
    plt.show()


'''
example:

d = [[0,772.11],[1,931.23],[2,931.23],[3,945.68],[4,945.68],[5,1075.96],[6,1115.61]]
get_linear_intercept(data)

output:
(47.657499999999985, 816.6703571428571)
'''
