import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing , svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

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


# features
X = np.array(df.drop(['label'], 1))
#  scalling the data
X = preprocessing.scale(X)
X = X[:-forcast_out]
# the values we wanna predict
X_lately = X[-forcast_out:]
df.dropna(inplace = True)
# label
Y = np.array(df['label'])

# split data to training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# creating a classifier and choosing an algorithm to use 
classifier = LinearRegression()
# using another algorithm (support vector regression)
# classifier = svm.SVR()
# train
classifier.fit(X_train, Y_train)

# pickle the classifier
with open('linearregression.pickl', 'wb') as f:
    pickle.dump(classifier, f)

pickle_in = open('linearregression.pickl', 'rb')
classifier = pickle.load(pickle_in)

# the accuracy of the predictions
accuracy = classifier.score(X_test, Y_test)
# predicted values
forcast_set = classifier.predict(X_lately)
df['Forcast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]


# y = mx + b

print('accuracy : ', accuracy)
print(df.head(37))

# for the graph representation
df['Adj. Close'].plot()
df['Forcast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()











