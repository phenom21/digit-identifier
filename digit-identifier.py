from sklearn.linear_model import LogisticRegression # Not used
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve

import pandas as pd
import matplotlib from matplotlib import pyplot as plt
import numpy as np

print('''If you are using Jupyter notebook, add this line -- "%matplotlib inline"\n''')

train_data = pd.read_csv("train.csv").iloc[ : 6000, : ].astype(np.int64)

print(train_data.info())

# Training the data and getting x_train and y_train
x_train, y_train = train_data.iloc[ : , 1 : ], train_data.iloc[ : , 0]
some_digit = np.array([x_train.iloc[301, :]])
img = some_digit.reshape(28, 28)

''' We have not used %matplotlib inline here because we are running this project in VS Code which will display a block figure in a separate window. Had it been Jupyter notebook we should have used it essentially '''
# %matplotlib inline

''' This statement converts a 2D numpy array into an image '''
plt.imshow(img, cmap = matplotlib.cm.binary , interpolation = "nearest")
mm = f"This is just a demo\nThis number is {y_train.iloc[301]}"

plt.title(mm)
plt.show()

# print("The number at 301st index is ",y_train.iloc[301])

# USER INPUT
us = int(input("\nEnter the number whose Analysis you want to see\n"))

test_data = pd.read_csv("test.csv").iloc[ : 5001, : ].astype(np.int64)

upd_y_train = (y_train == us).astype(np.int8)

x_test = test_data


clf = KNeighborsClassifier()

# Training our Classifier
clf.fit(x_train, upd_y_train)


''' Here, we can match the result of our model and the picture '''

some_dig = np.array([x_test.iloc[274, :]])
img = some_dig.reshape(28, 28)
plt.imshow(img)
# This if else condition is just for demo
nn = f"The number is {us}"
jj = f"The number is not {us}"
if clf.predict([x_test.iloc[274, : ]]) == 1:
    plt.title(nn)
else:
    plt.title(jj)
plt.show()


# we are using training data set for x and y variables here, because we are cross validating within the training data
# testing data has no concern with it
a = cross_val_score(clf, x_train, upd_y_train, cv = 3, scoring = "accuracy")
a.mean() # It will approximately come out to be 97% - 98%

''' We should be happy right(after all 97 % accuracy is not a joke)!!! But NOOOO!!
because a good data scientist will know that already there are only 10% " 3 " in the whole dataset becuase of probability.
Even if our model classifies all of them as False, then also 90% accuracy will be there.
Thus, accuracy is not a suitable PARAMETER for determining a model's performance '''

'''

Here comes CONFUSION MATRIX, Precision, Recall and F1 factor

'''

# cross_val_pred gives us the value of labels by cross validating them within the dataset

# upd_y_train is written here because training process needs to take values form labels,
# which in this case is provided by upd_y_train
y_train_pred = cross_val_predict(clf, x_train, upd_y_train, cv = 4)

conf_mat = confusion_matrix(upd_y_train, y_train_pred)
print("\n\nThe confusion matrix is as follows\n", conf_mat)

prec = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1])
print(f"\nThe precision of our model to find {us} is {prec}\n")

rec = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
print(f"The recall of our model to find {us} is {rec}\n")

f1 = 2 * prec * rec/ (prec + rec)
print(f"The f1 score of our model to find {us} is {f1}\n")


'''

This is a doubt that has come to me many times now. 

Why do we use x_train and y_train_pred to create 
Precision Recall curve, why not x_test and y_test_pred?

Because we have used x_train and y_train_pred to calculate the precision and recall of the dataset
HAPPY!!!

'''
plot_precision_recall_curve(clf, x_train, y_train_pred)
plt.title("Precision-Recall Curve")
plt.show()

'''

From here, the user can specify the test case[ranging from (0 to 5000)], 
simultaneously seeing the image and the result of the model.

'''
choice = "YES"
while choice == "YES" or choice == "Yes" or choice == "yes" or choice == 'y':
    pp = int(input("\nEnter the test case index[from 0 to 5000] "))

    to_be_displayed = np.array([x_test.iloc[pp, : ]])

    ima = to_be_displayed.reshape(28, 28)
    plt.imshow(ima)

    if clf.predict(to_be_displayed) == 1:
        plt.title(nn)
    else:
        plt.title(jj)

    plt.show()

    choice = input("Do you want to try for another value of index? ")
print("\n\nThat was all from this project. I hope it did well.")
