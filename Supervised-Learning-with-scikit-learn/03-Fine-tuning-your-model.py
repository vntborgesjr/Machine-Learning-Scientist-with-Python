# --------------------------------------------------- 
# Supervised learning wiwth scikit-learn - Fine-tuning your model 
# 28 jan 2021 
# VNTBJR 
# --------------------------------------------------- 
#

######################################################################
# How good is your model?  -------------------------------------------
######################################################################
# In case of class imbalance in classification models (e.g. one 
# class is more frequent) accuracy is not the best measure of model 
# efficience. Instead we use metrics derived from the confusion 
# matrix such as precision, recall (also called sensitivity, hit
# rate, or true positive rate), and F1-score.  
# High precision means low false positive rate.
# High recall means that our classifier predict most positive cases
# correctly.
# Metrics for classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('datasets/diabetes.csv')
df.info()
X = pd.DataFrame(df.drop('diabetes', axis = 1).values)
y = pd.Series(df['diabetes'].values)

# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size = 0.4, random_state = 42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

######################################################################
# Logistic regression and the ROC curve  -------------------------------------------
######################################################################
# Building a logistic regression model
# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plotting an ROC curve
# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
