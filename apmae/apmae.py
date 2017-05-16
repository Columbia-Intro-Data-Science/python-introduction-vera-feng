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

############################################################################

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
    #loop through all X columns and identify the categorical columns
    categoricals = []
    for col, col_type in df.dtypes.iteritems():
        if col_type == 'O':
        	categoricals.append(col)
        else:
            categoricals = categoricals
              
    #convert the categorical variables to dummy variables
    df_ohe = pd.get_dummies(df, columns=categoricals)
    
   
    #create X and y variables
    y = df_ohe["Deal status"]
    X = df_ohe.drop(['Deal status'],1)
    
    return (X, y)

def ask_for_user_input(filename, x1, x2, x3, x4, x5):

    #append the user input to the original dataframe
    user_input = [x1, x2, x3, x4, x5, "Completed"]
    
    df2 = pd.read_csv(filename, sep="\t")
    df2 = df2.drop(["Mark", "Deal Number", "Acquiror name", "Target name"],1)
    df2 = df2.dropna(how='any')
    df2.loc[len(df2)]= user_input
    
    df2["Deal status"] = df2["Deal status"].astype("category")
    df2["Deal status"] = df2["Deal status"].apply(convert_y)
    
    categoricals = []
    for col, col_type in df2.dtypes.iteritems():
         if col_type == 'O':
              categoricals.append(col)
         else:
              categoricals = categoricals

    df_ohe2 = pd.get_dummies(df2, columns=categoricals)
    
    
    #create X and y variables
    y2 = df_ohe2["Deal status"]
    X2 = df_ohe2.drop(['Deal status'],1)
    
    X_test = X2.tail(1)

    return X_test
    
def optimal_regression(input_data):
    #find out which depth of decision tree regression is optimal
    
    X = input_data[0]
    y = input_data[1]
    
    #split the dataset into training, validate & test datasets
    X_model, X_test, y_model, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    X_train, X_validate, y_train, y_validate = train_test_split(X_model, y_model, test_size=0.2, random_state=5)
    
    #find out the optimal tree depth
    #using train and validate datasets
    train_errors=[]
    validate_errors=[]
    scores=[]
    depths = range(1,6)
    for n in depths:
        clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=n)
        # Train the model using the training sets
        clf_entropy.fit(X_train, y_train)
        train_errors.append(clf_entropy.score(X_train,y_train))
        scores.append(clf_entropy.score(X_validate,y_validate))
    plt.ylabel('R^2')
    plt.xlabel('Depth')
    plt.plot(depths,scores)
    validate_errors=scores
    n_opt=depths[np.argmax(scores)]
    
    #regr_opt = DecisionTreeRegressor(max_depth=n_opt)
    #regr_opt.fit(X_model, y_model)
    clf_entropy_opt = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=n_opt)
    clf_entropy_opt.fit(X_model, y_model)
    
    y_pred = clf_entropy_opt.predict(X_test)
    
    #performance measure of the optimal model
    #percentage of correct predictions
    #using test dataset
    
    counter = 0
    for y_p, y_t in zip(y_pred, y_test):
        if y_p == y_t:
            counter += 1
        else:
            counter = counter
    accuracy = 100*counter/len(y_test)
    
    return (n_opt, accuracy)
    
def regression_and_prediction(input_data, optimal_n, X_test):
    """this function uses the optimal decision tree regression to make the best prediction
    """
    
    #our dataset
    X = input_data[0]
    y = input_data[1]
    
    #identify the class
    clf_entropy_opt = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=optimal_n)
    clf_entropy_opt.fit(X, y)
    y_pred = clf_entropy_opt.predict(X_test)
    
    #run the regression to identify the probability
    regr = DecisionTreeRegressor(max_depth=optimal_n)
    regr.fit(X, y)
    y_prob = regr.predict(X_test)
    
    return (y_pred, y_prob)
    
####################################################################  

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("deal.html")
    
@app.route('/', methods=['GET', 'POST'])
def my_form_post():

    x1 = request.form.get("deal type")
    x2 = request.form.get("payment method")
    x3 = request.form["fstake"]
    x3 = format(int(x3), '.5f')
    x3 = str(x3)
    x4 = request.form.get("acquiror major sector")
    x5 = request.form.get("target major sector")
	
	# Read file, return tuple
    input_data = input_file(datafile)
    print(input_data[0].shape)
	
	# Train decision tree, return optimal depth and its accuracy
    regression = optimal_regression(input_data)
    n_opt= regression[0]
    accuracy = regression[1]
	
	# grab and process user input 
    X_test = ask_for_user_input(datafile, x1, x2, x3, x4, x5)
    
    # predict and return prediction and its corresponding probability
    prediction = regression_and_prediction(input_data, n_opt, X_test)
    
    #relevants result stats
    prob = str(prediction[1][0])
    
    #show the results
    if prediction[0] == 1:
        return render_template("positive.html", prob = prob, n_opt = n_opt, accuracy = accuracy)
    else:
        return render_template("negative.html", prob = prob, n_opt = n_opt, accuracy = accuracy)


if __name__ == '__main__':
    app.run()


