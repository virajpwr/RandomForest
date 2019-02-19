
# Import dependencies
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


# Import data
data = pd.read_csv('winequality_dataset.csv')
print(data.columns)
X_data = data[['fixed acidity', 'citric acid', 'sulphates', 'alcohol']]

# Create output classes 
reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('Bad')
    elif i >= 4 and i <= 7:
        reviews.append('Average')
    elif i >= 8 and i <= 10:
        reviews.append('Good')
data['WineQuality'] = reviews

X = X_data
y = data['WineQuality']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Train Model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)

Accuracy = accuracy_score(y_test, rf_predict)

print('Accuracy of the model: {}%'.format(Accuracy*100))

joblib.dump(rf,"RandomForest.pkl", protocol =2) # 
model_columns = list(X_data.columns)
joblib.dump(model_columns, "model_columns.pkl", protocol =2)
print("Model columns dumped")