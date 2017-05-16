from flask import Flask
from flask import request
from flask import render_template

import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Source
datafile = "data.txt"

##########################################
def convert_y(variable):
    #this function is to convert y to a dummy variable
    #if y= "completed" --> y =1
    #else: y=0
    
    if "Completed" in variable:
        variable = 1
    else:
        variable = 0
        
    return variable

def input_file(filename):
    
    """This function will do the following:
    - read a datafile
    - process and convert the data
    - create X's and Y variables
    - store the results
    """
    
    #read the datafile 
    #and extract relevant data
    df = pd.read_csv(filename, sep="\t")
    df = df.drop(["Mark", "Deal Number", "Acquiror name", "Target name"],1)
    df = df.dropna(axis=0, how='any')

    #convert the y variable to binary
    df["Deal status"] = df["Deal status"].astype("category")
    df["Deal status"] = df["Deal status"].apply(convert_y)

    #convert categorical X variables to numerical values
    categoricals = []
    for col, col_type in df.dtypes.iteritems():
         if col_type == 'O':
              categoricals.append(col)
         else:
              df[col].fillna(0, inplace=True)

    df_ohe = pd.get_dummies(df, columns=categoricals)
    
    #create X and y variables
    y = df_ohe["Deal status"]
    X = df_ohe.drop(['Deal status'],1)
    
    return (X, y)
    
    
###########################################    

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("deal.html")
    
@app.route('/', methods=['POST'])
def my_form_post():
	
	# Read file, return tuple
	input_data = input_file(datafile)
	print(input_data[0].shape)
	
	x3 = request.form["fstake"]
	print(x3)
	return x3


if __name__ == '__main__':
    app.run()


