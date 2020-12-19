import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing , svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split

df = quandl.get('WIKI/GOOGL')
# features
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PRCNT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0

# features
df = df[['Adj. Close', 'HL_PRCNT', 'PCT_Change', 'Adj. Volume']]
# we use this col to predict the closing price for the next day or days 
forcast_col = 'Adj. Close'
# fill non available data (in pandas its NANs not a number)
df.fillna(-99999, inplace = True)
# predict 10% out of the data frame (number of days to predict)
forcast_out = int(math.ceil(0.01*len(df)))
# the predicted closing price
df['label'] = df[forcast_col].shift(-forcast_out)
df.dropna(inplace = True)

# features
X = np.array(df.drop(['label'], 1))
# label
Y = np.array(df['label'])
#  scalling the data
X = preprocessing.scale(X)

# split data to training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# creating a classifier and choosing an algorithm to use 
classifier = LinearRegression()
# using another algorithm (support vector regression)
# classifier = svm.SVR()
# train
classifier.fit(X_train, Y_train)
# the accuracy of the predictions
accuracy = classifier.score(X_test, Y_test)




print('accuracy : ', accuracy)
print(df.head())

















