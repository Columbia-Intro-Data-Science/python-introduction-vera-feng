import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    # Source
    datafile = "data.txt"
    
    # Read file, return tuple
    input_data = input_file(datafile)
    
    # Train decision tree, return optimal depth
    optimal_n = optimal_regression(input_data)

    # get user input 
    X_test = ask_for_user_input(datafile)

    # predict
    prediction = regression_and_prediction(input_data, optimal_n, X_test)  

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


def ask_for_user_input(filename):
    
    """This function asks the user to enter the values of X variables as the test data"""
    """
    #give the prompts
    print("To assist you make an educated guess of the success rate of your M&A deal")
    print("I need some information from you")

    #ask for values of the predictor variables
    #the first variable: deal type
    MSG = ["What's the type of your deal?\n
    		Please choose one from below
    		",
    
    ]
    
    print("What's the type of your deal?\n")
    print("Please choose one from below")
    print("Acquisition, Institutional buy-out, Joint-venture, MBI / MBO, Management buy-in, Management buy-out, Merger")
    x1_test = input()
    #x1_test = x1_test.lower()
    print("\n")
    #the second variable: the method of payment
    print("What's the method of payment?")
    print("Please choose one from below")
    print("Cash, Cash assumed, Converted Debt, Debt assumed, Deferred payment, Earn-out, Loan notes, Other, Shares")
    x2_test = input()
    #x2_test = x2_test.lower()
    print("\n")
    #the third variable: final stake
    print("How much final stake would you hold in the target company after this M&A?")
    print("Please enter a percentage point between 0 to 100")
    x3_test = input()
    #x3_test = float(x3_test)
    print("\n")
    #the fourth variable: acqurior major sector
    print("What's your major sector as the acquiror?")
    print("Please choose one from below")
    print("Primary Sector (agriculture, mining, etc.), Food, beverages, tobacco, Textiles, wearing apparel, leather, Wood, cork, paper, Publishing, printing, Chemicals, rubber, plastics, non-metallic products, Metals&metal products, Machinery, equipment, furniture, recycling, Gas, Water, Electricity, Construction, Wholesale&retail trade, Hotels&restaurants, Transport, Post and telecommunications, Banks, Insurance companies, Other services, Public administration and defence, Education, Health")
    x4_test = input()
    #x4_test = x4_test.lower()
    print("\n")
    #the fifth variable: target company sector
    print("What's the target company's major sector?")
    print("Please choose one from below")
    print("Primary Sector (agriculture, mining, etc.), Food, beverages, tobacco, Textiles, wearing apparel, leather, Wood, cork, paper, Publishing, printing, Chemicals, rubber, plastics, non-metallic products, Metals&metal products, Machinery, equipment, furniture, recycling, Gas, Water, Electricity, Construction, Wholesale&retail trade, Hotels&restaurants, Transport, Post and telecommunications, Banks, Insurance companies, Other services, Public administration and defence, Education, Health")
    x5_test = input()
    
    print("\n")

    #store the input values
    print("A summary of your M&A deal info:")
    print("Your deal type is", x1_test)
    print("Your method of payment is", x2_test)
    print("Your final stake is " + str(x3_test) + "%")
    print("You are in " + x4_test + " industry")
    print("The target company is in " + x5_test + " industry")

    #convert the user input into the dummy variables to match the regression
    user_input = [x1_test, x2_test, x3_test, x4_test, x5_test, "Completed"]
    """
    #append the user input to the original dataframe
    user_input = ["Acquisition 100%", "Shares", "100.00000", "Post and telecommunications", "Other services", "Completed"]
    
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
    print('The optimal depth is ' + str(n_opt))
    
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
    performance = 100*counter/len(y_test)
    print("The prediction accuracy is " + str(performance) + "%")
    
    return n_opt

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
    
    #print the results
    if y_pred == 1:
        print("We predict that your M&A will be successfully completed with a probability of " + str(y_prob[0]))
    else:
        print("Unfortunately the prospect of your M&A is not looking rosy. The probability of successfully concluding your deal is only " + str(y_prob[0]))
    
    return y_pred

main()