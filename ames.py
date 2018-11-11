"""
Universidad Simon Bolivar
Artificial Intelligence II - CI5438
Authors:
    David Cabeza 1310191
    Rafael Blanco 1310156
    Fabiola Martinez 1310838
"""

import numpy as np
import pandas as pd
from project_1 import * # contains linear regression algorithm and his functions.
from plots import * # contains tools for graphics.

def get_samples_DeCock(df, cols, force=True):
    """
    Return training and validation data samples using pandas's
    random sampling and Dean De Cock suggested method.

    De Cock states the following:

    "The two data sets can be easily created by randomizing the original data 
    and selecting the relevant proportion for each component with the only real
    requirement being that the number of observations in the training set be six 
    to ten times the number of variables."
    
    when force=True the training dataset lower limit is six times the number
    of variables (included), when false, the lower limit is the number
    of variables.

    NOTE: This function could be improved by relaxing the min_validation_size within an 
    interval.
    """
    max_factor = 10
    min_factor = 6

    # calculate number of rows
    rows = len(df)

    # this algorithm begins trying with max factor and substracts one until
    # it finds the mentioned proportion of training/validation data.
    # Always makes sure validation data size is not less than 20% of the data.    
    factor = max_factor
    min_validation_size = round(rows/5)

    if min_validation_size == 0:
        # dont waste your time.
        raise ValueError("Dataset too small.")

    while (force and factor >= min_factor) or (factor > 0):
        training_size = cols*factor
        validation_size = rows - training_size
        
        if (validation_size >= min_validation_size):
            df_validation = df.sample(n=validation_size)
            df_training = df.sample(n=training_size)
            return df_training, df_validation
        else:
            factor -= 1
            continue
    
    raise ValueError("Dataset too small.")


def fix_missing_with_mode(df):
    """
    Fixes missing value from all columns using the mode.
    """
    return df.fillna(df.mode().iloc[0])

def only_rows_from_numeric_gte_column_value(df, column, value):
    """
    Remove from dataframe rows whose 'column' value is not >= 'value'
    """
    return df.loc[df[column] >= value]

def drop_column(df, column):
    """
    Drop from dataframe 'column'
    """
    try:
        df = df.drop(column, axis=1)
    except ValueError as err:
        print(err)

    return df

def only_rows_from_categorical_column_value(df, column, value):
    """
    Remove rows from dataframe.

    Remaining dataframe will have ONLY rows whose column 'column' 
    has value 'value'
    """
    return df.loc[df[column] == value]

def dummies(df):
    """
    Get dummies variables for categorical data.
    """
    return pd.get_dummies(df)

def read_file(filename):
    """
    Reads file and process it using panda dataframes.
    
    @param name of the file
    @return dataframe
    """
    try:
        df = pd.read_csv(filename)
        return df
    except:
        print("File couldn't be read")
        sys.exit(-1)

"""
Description: gets information about two datasets.

Parameters:
    @param filename1: name of de dataset file.
    @param filename2: name of de dataset file.
"""
def read_dataset(filename1, filename2):
    data_1 = []
    data_2 = []
    k = 0

    data_training = open(filename1, "r")
    data_validation = open(filename2, "r")

    for line in data_training:
        if k == 0:
            k+=1
            continue
        word = line.split(",")
        word[0]=1  
        # data_1.append(1)
        data_1.append(word[0:len(word)-1])

    k = 0

    for line in data_validation:
        if k == 0:
            k+=1
            continue
        word = line.split(",")
        word[0]=1
        data_2.append(word[0:len(word)-1])

    for i in range(len(data_1)):
        for j in range(len(data_1[i])):
            data_1[i][j] = float(data_1[i][j])

    for i in range(len(data_2)):
        for j in range(len(data_2[i])):
            data_2[i][j] = float(data_2[i][j])

    return data_1, data_2

"""
Description: normalizes datasets.

Parameters:
    @param data1: dataset to be normalized.
    @param data2: dataset to be normalized.
"""
def norm_ames(data1, data2):
    media=[1]
    varianza=[1]
    
    for i in range(1,len(data1[0])):
        aux=0
        for j in range(len(data1)):
            aux+=float(data1[j][i])

        for j in range(len(data2)):
            aux+=float(data2[j][i])
        
        media.append(aux/len(data1+data2))
    
    for i in range(1,len(data1[0])):
        aux=0
        for j in range(len(data1)):
            aux+=(float(data1[j][i])-media[i])**2

        for j in range(len(data2)):
            aux+=(float(data2[j][i])-media[i])**2
        varianza.append((aux/(len(data2+data1)-1))**(1/2))
    
    for i in range(1,len(data1[0])):
        for j in range(len(data1)):
            if varianza[i] != 0:
                data1[j][i]=(float(data1[j][i])-media[i])/varianza[i]
        for j in range(len(data2)):
            if varianza[i] != 0:
                data2[j][i]=(float(data2[j][i])-media[i])/varianza[i]

    return data1, data2


"""
Description: calculates the mean.

Parameters:
    @param x: x values.
"""
def mean(x):
    acum=0

    for i in range(len(x)):
        acum+=x[i]

    return acum/len(x)

"""
    Description: subtracts two vectors and return absolute value.

    Parameters:
        @param a: a vector.
        @param b: a vector.
"""
def sub_vec_abs(a,b):
    
    c=[]
    for i in range (0,len(a)):
        c.append(abs(a[i]-b[i]))
    return c

"""
    Description: applies jfunc.

    Parameters:
        @param x: x values.
        @param y: y values.
"""
def jota_validation(thetas,x,y):
    aux=[]

    for i in range(len(thetas)):
        aux.append(jfunc(thetas[i],x,y))

    return aux

"""
Description: main.
"""
def init():
    df = read_file("ww2.amstat.org.txt")

    # a) Data cleaning
    df = only_rows_from_categorical_column_value(df, "Sale Condition", "Normal")
    df = only_rows_from_numeric_gte_column_value(df, "Gr Liv Area", 1500)
    
    # other operations
    df = drop_column(df, 'PID')
    df = drop_column(df, 'Order')
    df = fix_missing_with_mode(df)

    # calculate number of columns for sampling before getting dummies 
    # with dummies, there will be more variables so the sets would need
    # to be extremely large.
    cols = len(df.columns)
    
    df = dummies(df)

    # c) Data splitting
    df_training, df_validation = get_samples_DeCock(df, cols)

    df_training.to_csv("amstat_training.txt")
    df_validation.to_csv("amstat_validation.txt")

    # b) Data normalization    
    training, validation = read_dataset("amstat_training.txt", "amstat_validation.txt")
        
    training_y = []
    validation_y = []
    
    # Gets vector y of training data
    for i in range(len(training)):
        training_y.append(training[i][36])
        del(training[i][36])

    # Gets vector y of validation data
    for i in range(len(validation)):
        validation_y.append(validation[i][36])
        del(validation[i][36])

    # Normalizes the data
    training_norm, validation_norm  = norm_ames(training,validation)

    results_training, jota_t, thetas = gradient_descent(0.1,training_norm,training_y,5)
    
    # Comparation between mean square error of validation data and training data
    jota_valid = jota_validation(thetas,validation_norm,validation_y)
    iterations = np.arange(len(jota_t))
    iterations2 = np.arange(len(jota_valid))
    
    # Plotting
    plotNormal(iterations, jota_t,"Iteraciones", "J()", "Curva de Convergencia","#0174DF")
    plotNormal(iterations2, jota_valid,"Iteraciones", "J()", "Curva de Convergencia","#6A0888")

    legend1 = mpatches.Patch(color="#0174DF",label="Datos de entrenamiento")
    legend2 = mpatches.Patch(color="#6A0888",label="Error cuadratico medio de los datos de validacion")
    plt.legend(handles=[legend1, legend2])
    
    # d) Model Assesing under training and validation data

    # criteria: bias -- average(yhat - y)

    aprox_y = []

    for i in range(len(validation_norm)):
        aprox_y.append(h(results_training,validation_norm[i]))

    bias = mean(sub_vec(aprox_y,validation_y))
    print("Bias: ",bias)

    # criteria: maximum deviation -- max(|y-yhat|)

    max_deviation = max(sub_vec_abs(validation_y,aprox_y))
    print("Maximum Deviation: ",max_deviation)

    # criteria: mean absolute deviation -- average(|y-yhat|)

    mean_abs_deviation = mean(sub_vec_abs(validation_y,aprox_y))
    print("Mean absolute deviation: ",mean_abs_deviation)

    # criteria: mean square error -- average(y-yhat)^2

    mean_sq_error = mean(np.power(sub_vec(validation_y,aprox_y),2))
    print("Mean Square Error: ",mean_sq_error)

    plt.show()

if __name__ == '__main__':
    init()
