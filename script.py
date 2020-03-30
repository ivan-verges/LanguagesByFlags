import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#Loads Flags data from dataset into a Dataframe
flags = pd.read_csv("flags.csv", header = 0)

#Load the interested features from Dataframe
data  = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange", "Circles", "Crosses", "Saltires", "Quarters", "Sunstars", "Crescent", "Triangle"]]

#Load the interested label to predict from Dataframe
labels = flags["Language"]

#Splits data into Train Data, Test Data, Train Labels and Test Labels
train_data, test_data, train_labels, test_labels = train_test_split(data, labels)

#Tests Decision Tree Model with many depth and finds th best one
scores = []
for i in range(1, 21):
  tree = DecisionTreeClassifier(max_depth = i)
  tree.fit(train_data, train_labels)
  scores.append(tree.score(test_data, test_labels))

#Shows a Graph about the precision of the model as it's depth changes
plt.plot(range(1, 21), scores)
plt.show()

#Prints the prediction for the selected flag
print(tree.predict(test_data[0:1]))