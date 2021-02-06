# --------------------------------------------------- 
# Supervised learning wiwth scikit-learn - Preprocessing and pipelines 
# 27 jan 2021 
# VNTBJR 
# --------------------------------------------------- 
#

######################################################################
# Preprocessing data  -------------------------------------------
######################################################################
# Exploring categorical features
import matplotlib.pyplot as plt

# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('datasets/gm_2008_region.csv', sep = ',')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()

# Creating dummy variables
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first = True)

# Print the new columns of df_region
print(df_region.columns)

# Regression with categorical features
# Import necessary modules
____
____

# Instantiate a ridge regressor: ridge
ridge = ____

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = ____

# Print the cross-validated scores
print(____)


# convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

# Steps to building a model:
# 1 - create training and test sets
# 2 - fit a classifier or regressor
# 3 - tuning its parameters
# 4 - evaluating its performance on new data

# Imputing missing data in a ML pipeline I
