import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset 
dataset = pd.read_csv('Data.csv')

# take all the columns except last one
X = dataset.iloc[:,:-1].values
# take the last columns
y = dataset.iloc[:,3].values

# missing data 
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(X_train)
x_test = sc_X.transform(X_test)
