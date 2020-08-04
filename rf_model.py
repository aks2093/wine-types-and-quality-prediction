import numpy as np
import pandas as pd
from random import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

np.random.seed(0)

data = pd.read_csv(r"C:\Users\amis\PycharmProjects\wines-type-and-quality-classification\training_data\wine_train.csv")
data['quality'] = data.quality.apply(lambda q: 0 if q <= 5 else 1 if q == 6 else 2)

mapping = {0:"bad",1:"normal",2:"good"}

x = ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar',
       'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

target = ["quality"]

# data = shuffle(data)

data = data.sample(frac=1).reset_index(drop=True)

X = data[x].values
Y = data[target].values

standard_sc = StandardScaler()
X = standard_sc.fit_transform(X)

X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y)

rfc = RandomForestClassifier(n_estimators=2000, max_depth=20, min_samples_split=4)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_test)
print("AccuracyScore: ", accuracy_score(predictions, Y_test))
print("recall_score: ", recall_score(predictions, Y_test, average='macro'))
print("precision_score: ", precision_score(predictions, Y_test, average='macro'))



