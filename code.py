# Load packages
import numpy as np 
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Load data from the csv file
df = pd.read_csv("housing.csv", index_col=False)
df.head()

CONTAMINATION=.2    # The expected outliers to real data ratio
BOOTSTRAP = False   # True if you want to use the bootstrap method; 

# Exctract the data without column names from the dataframe
data = df.values

# We will start by splitting the data into X/y train/test data
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Next, we will set up the Isolation Forest classifier that will detect the outliers
i_forest = IsolationForest(contamination=CONTAMINATION, bootstrap=BOOTSTRAP)
is_inlier = i_forest.fit_predict(X_train)    # +1 if inlier, -1 if outlier

# Finally, we will select the rows without outliers
mask = is_inlier != -1
# and remove these from the train data
X_train, y_train = X_train[mask, :], y_train[mask]

df_wo_outliers=pd.DataFrame(X_train, columns=df.columns[:-1])
df_boxplot = pd.DataFrame(data={'With outliers':df['RM'], 'Without outliers':df_wo_outliers['RM']})
fig = px.box(df_boxplot)
fig.show()