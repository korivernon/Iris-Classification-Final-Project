# Kori Vernon
# Final Project
# Iris Automatic Classification

import math
import turtle

def create_table(file_name):
    """
    sig: str -> tuple(list(float), list(float), list(str))
    Given a file name, read the file into a tuple containing
    two lists of type float and one list of type string.
    The features of the dataset should be of type float
    and the label should be of type string. 
    """
    length_of_iris = []
    height_of_iris = []
    name_of_iris = []
    iriscsv = open(file_name, 'r')
    for line in iriscsv: 
        row = line.split(',')
        height_of_iris.append(float((row[0])))
        length_of_iris.append(float((row[1])))
        name_of_iris.append(str(row[2]).rstrip('\n'))
        tup = (height_of_iris,length_of_iris,name_of_iris)
    return tup
    
def print_range_max_min(data):
    """
    sig: tuple(list(float), list(float)) -> NoneType
    Print the max, min and range of both features in the dataset.
    """
    max_ht = 0
    min_ht = 10 
    max_ln = 0
    min_ln = 10
    for i in data[0]:
        # Go through whatever the two lists are... 
        # Height Max
        if max_ht > i:
            max_ht = max_ht
        else:
            max_ht = i
        # Height Minimum
        if min_ht < i:
            min_ht = min_ht
        else:
            min_ht = i
    for i in data[1]:
         # Length Max
        if max_ln > i:
            max_ln = max_ln
        else:
            max_ln = i
        # Length Minimum
        if min_ln < i:
            min_ln = min_ln
        else:
            min_ln = i
    print("Feature 1 - Max:", max_ht,"Min:", min_ht, "range:", (max_ht - min_ht))
    print("Feature 2 - Max:", max_ln,"Min:", min_ln, "range:", (max_ln - min_ln))

def find_mean(feature):
    mean = 0
    for i in range(len(feature)):
        mean += feature[i]
    mean = mean*1.0/len(feature)
    return mean

def find_std_dev(feature, mean):
    sum_iris = 0
    total_sum = 0 
    for i in range(len(feature)):
        total_sum += (feature[i] - mean)**2
        under_root = total_sum*1.0/len(feature)
    return math.sqrt(under_root)


def normalize_data(data):
    """                                                                                                                                                                           
    sig: tuple(list(float), list(float), list(str)) -> NoneType                                                                                                                   
    Print the mean and standard deviation for each feature.                                                                                                                       
    Normalize the features in the dataset by                                                                                                                                      
    rescaling all the values in a particular feature                                                                                                                              
    in terms of a mean of 0 and a standard deviation of 1.                                                                                                                        
    Print the mean and the standard deviation for each feature, now normalized.                                                                                                   
    After normalization, each of your features should display a mean of 0                                                                                                         
    or very close to 0 and a standard deviation of 1 or very close to 1.                                                                                                          
    """
    mean_ln = find_mean(data[1])
    mean_ht = find_mean(data[0])
    std_ln = find_std_dev(data[1], mean_ln)
    std_ht = find_std_dev(data[0], mean_ht)
    
    for i in range(len(data[0])):
        ht_norm = (data[0][i] - mean_ht) / std_ht
        data[0][i] = ht_norm
        ln_norm = (data[1][i] - mean_ln) / std_ln
        data[1][i] = ln_norm

    norm_ln_mean = find_mean(data[1])
    norm_ln_std = find_std_dev(data[1], norm_ln_mean)

    norm_ht_mean = find_mean(data[0])
    norm_ht_std = find_std_dev(data[0], norm_ht_mean)

    print("Feature 1 - mean:" , mean_ht, "std dev:", std_ht)
    print("Feature 1 after normalization mean:", norm_ht_mean, "std dev:", norm_ht_std )
    print("Feature 2 - mean:" , mean_ln, "std dev:", std_ln)
    print("Feature 2 after normalization mean:", norm_ln_mean, "std dev:", norm_ln_std )

def make_predictions(train_set, test_set):
    """
    sig: tuple(list(float), list(float), list(str)), tuple(list(float), list(float), list(str)) -> list(str)
    For each observation in the test set, you'll need to check all of
    the observations in the training set to see which is the `nearest neighbor.'
    The function should make a call to the function find_dist.
    Accumulate a list of predicted iris types for each of the test set
    observations. Return this prediction list.
    """
    n = len(test_set)
    position = 1
    predicted_names = []
    for i in range(len(test_set[1])):
        distance_comparison = float('inf')
        for j in range(len(train_set[1])):
            x1 = test_set[0][i]
            y1 = test_set[1][i]
            x2 = train_set[0][j]
            y2 = train_set[1][j]
            distance = find_dist(x1,y1,x2,y2)                     
            if distance <= distance_comparison:
                distance_comparison = distance 
                position = train_set[2][j]
        predicted_names.append(position)
            
    return predicted_names #return pred_lst
            
def find_dist(x1, y1, x2, y2):
    """
    sig: float, float, float, float -> float
    Return the Euclidean distance between two points (x1, y1), (x2, y2).
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
def find_error(test_data, pred_lst):
    """
    sig: tuple(list(float), list(float), list(str)) -> float
    Check the prediction list against the actual labels for
    the test set to determine how many errors were made.
    Return a percentage of how many observations in the
    test set were predicted incorrectly. 
    """
    percentage = 0
    total = len(test_data[2])
    count = 0
    for i in range(total):
        if test_data[2][i] != pred_lst[i]:
            count+=1
        else:
            count += 0
    percentage = float(count/total)
    return percentage

def plot_data(train_data, test_data, pred_lst):
    """
    sig: tuple(list(float), list(float), list(str)), tuple(list(float), list(float), list(str)), list(str)
        -> NoneType
    Plot the results using the turtle module. Set the turtle window size to 500 x 500.
    Draw the x and y axes in the window. Label the axes "petal width" and "petal length". 
    Plot each observation from your training set on the plane, using a circle shape
    and a different color for each type of iris. Use the value of the first feature
    for the x-coordinate and the value of the second feature for the y-coordinate.
    Use a dot size of 10. Recall that the features have been normalized to have a mean
    of 0 and a standard deviation of 1. You will need to `stretch' your features across
    the axes to make the best use of the 500 x 500 window. Ensure that none of your
    points are plotted off screen. Also plot each correct prediction from your test
    set in the corresponding color. Use a square to indicate that the value is a prediction.
    Plot the incorrect predictions that were made for the test set in red, also using a
    square to indicate that it was a prediction. Include a key in the upper left
    corner of the plot as shown in the sample plot. The function will make a call
    to the function draw_key in order to accomplish this task. 
    """
    turtle.tracer(0,0)
    turtle.setup(width=500,height=500)
    
    # Plots X Axis
    turtle.setworldcoordinates(0,0,500,500)
    turtle.pu()
    turtle.setpos(0,250)
    turtle.pd()
    turtle.forward(500)
    # Writes Petal Length
    turtle.pu()
    turtle.setpos(400,250)
    turtle.pd()
    turtle.write("Petal Length", font=("Times New Roman", 18 , "normal"))
    turtle.pu()
    # Writes Petal Width
    turtle.setpos(255,0)
    turtle.pd()
    turtle.write("Petal Width", font=("Times New Roman", 18 , "normal"))
    turtle.pu()
    # Plots Y Axis
    turtle.setpos(250,0)
    turtle.left(90)
    turtle.pd()
    turtle.forward(500)
    turtle.pu()

    # Legend
    draw_key()
        
    for i in range(len(train_data[2])):
        # Iris-virinica
        if train_data[2][i] == 'Iris-virginica':
            turtle.pu()
            turtle.setpos((train_data[0][i]*120)+250,(train_data[1][i]*120)+250)
            turtle.pd()
            turtle.dot(10,"blue")
            turtle.pu()
        # Iris-versicolor
        elif train_data[2][i] == 'Iris-versicolor':
            turtle.pu()
            turtle.setpos((train_data[0][i]*120)+250,(train_data[1][i]*120)+250)
            turtle.pd()
            turtle.dot(10,"orange")
            turtle.pu()
        # Iris-setosa
        elif train_data[2][i] == 'Iris-setosa':
            turtle.pu()
            turtle.setpos((train_data[0][i]*120)+250,(train_data[1][i]*120)+250)
            turtle.pd()
            turtle.dot(10,"green")
            turtle.pu()      
    for i in range(len(pred_lst)):
        # Predicted Iris-virinica
        if pred_lst[i] == 'Iris-virginica' and pred_lst[i] == test_data[2][i]:
            turtle.pu()
            turtle.setpos((test_data[0][i]*120)+255,((test_data[1][i]*120)+255))
            turtle.color("blue", "blue")
            turtle.pd()
            turtle.begin_fill()
            for i in range(4):
                turtle.forward(10)
                turtle.left(90)
            turtle.end_fill()
        # Predicted Iris-veriscolor 
        elif pred_lst[i] == 'Iris-versicolor' and pred_lst[i] == test_data[2][i]:
            turtle.pu()
            turtle.setpos(((test_data[0][i]*120)+255,((test_data[1][i]*120)+255)))
            turtle.color("orange", "orange")
            turtle.pd()
            turtle.begin_fill()
            for i in range(4):
                turtle.forward(10)
                turtle.left(90)
            turtle.end_fill()
        # Predicted Iris-setosa
        elif pred_lst[i] == 'Iris-setosa' and pred_lst[i] == test_data[2][i]:
            turtle.pu()
            turtle.setpos(((test_data[0][i]*120)+255,((test_data[1][i]*120)+255)))
            turtle.color("green", "green")
            turtle.pd()
            turtle.begin_fill()
            for i in range(4):
                turtle.forward(10)
                turtle.left(90)
            turtle.end_fill()
        # Error
        elif pred_lst[i] != test_data[2][i]:
            turtle.pu()
            turtle.setpos(((test_data[0][i]*120)+255,((test_data[1][i]*120)+255)))
            turtle.color("red", "red")
            turtle.pd()
            turtle.begin_fill()
            for i in range(4):
                turtle.forward(10)
                turtle.left(90)
            turtle.end_fill()
    turtle.update()
    
def draw_key():
    """
    sig: () -> NoneType
    Draw the legend for the plot indicating which group is shown by each color/shape combination.  
    """
    # Legend Square
    turtle.pu()
    turtle.setpos(20,390)
    turtle.pd()
    turtle.forward(105)
    turtle.right(90)
    turtle.forward(140)
    turtle.right(90)
    turtle.forward(105)
    turtle.right(90)
    turtle.forward(140)
    # Legend Title
    turtle.pu()
    turtle.setpos(60, 473)
    turtle.pd()
    turtle.write("Legend", font=("Arial", 18 , "bold"))
    turtle.pu()
    
    # Blue Iris-virginica
    turtle.pu()
    turtle.setpos(30,468)
    turtle.pd()
    turtle.dot(10,"blue")
    turtle.pu()
    turtle.setpos(40, 460)
    turtle.pd()
    turtle.write("Iris-virginica", font=("Arial", 10 , "normal"))
    turtle.pu()
    # Orange Iris-veriscolor
    turtle.setpos(30,456)
    turtle.pd()
    turtle.dot(10,"orange")
    turtle.pu()
    turtle.setpos(40, 450)
    turtle.pd()
    turtle.write("Iris-versicolor", font=("Arial", 10 , "normal"))
    turtle.pu()
    # Green Iris-veriscolor
    turtle.setpos(30,445)
    turtle.pd()
    turtle.dot(10,"green")
    turtle.pu()
    turtle.setpos(40, 440)
    turtle.pd()
    turtle.write("Iris-setosa", font=("Arial", 10 , "normal"))
    turtle.pu()
    # Predicted Iris-virginica
    turtle.setpos(40, 428)
    turtle.pd()
    turtle.write("Predicted Iris-virginica", font=("Arial", 10 , "normal"))
    turtle.pu()
    # Predicted Iris-veriscolor
    turtle.setpos(40, 418)
    turtle.pd()
    turtle.write("Predicted Iris-veriscolor", font=("Arial", 10 , "normal"))
    turtle.pu()
    # Predicted Iris-setosa
    turtle.setpos(40, 408)
    turtle.pd()
    turtle.write("Predicted Iris-setosa", font=("Arial", 10 , "normal"))
    turtle.pu()
    # Predicted Incorrectly
    turtle.setpos(40, 398)
    turtle.pd()
    turtle.write("Predicted Incorrectly", font=("Arial", 10 , "normal"))
    turtle.pu()
    
    # Predicted Iris-virginica Square 
    turtle.setpos(33, 437.5)
    turtle.color("blue", "blue")
    turtle.pd()
    turtle.begin_fill()
    for i in range(4):
        turtle.forward(7)
        turtle.left(90)
    turtle.end_fill()
    turtle.pu()
    # Predicted Iris-virginica Square
    turtle.setpos(33, 427.5)
    turtle.color("orange", "orange")
    turtle.pd()
    turtle.begin_fill()
    for i in range(4):
        turtle.forward(7)
        turtle.left(90)
    turtle.end_fill()
    turtle.pu()
    # Predicted Iris-setosa Square
    turtle.setpos(33, 417.5)
    turtle.color("green", "green")
    turtle.pd()
    turtle.begin_fill()
    for i in range(4):
        turtle.forward(7)
        turtle.left(90)
    turtle.end_fill()
    turtle.pu()
    # Predicted Iris-setosa Square
    turtle.setpos(33, 407.5)
    turtle.color("red", "red")
    turtle.pd()
    turtle.begin_fill()
    for i in range(4):
        turtle.forward(7)
        turtle.left(90)
    turtle.end_fill()
    turtle.pu()

def main():
    """
    sig: () -> NoneType
    The main body of the program. It will use the other
    functions to load the data, process the training set,
    analyze the test set, and display its conclusions.
    """
    x = 0
    y = 0
    train_data = create_table("iris_train.csv")
    print_range_max_min(train_data[:2])
    print()
    normalize_data(train_data)
    test_data = create_table("iris_test.csv")
    print()
    normalize_data(test_data)
    pred_lst = make_predictions(train_data, test_data)
    error = find_error(test_data, pred_lst)
    print()
    print("The error percentage is: ", error)
    plot_data(train_data, test_data, pred_lst)

main()
