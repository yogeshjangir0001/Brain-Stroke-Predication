import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib


data = pd.read_csv('brain_stroke1.csv')

# Convert non-numerical data to numerical data
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['ever_married'] = le.fit_transform(data['ever_married'])
data['work_type'] = le.fit_transform(data['work_type'])
data['Residence_type'] = le.fit_transform(data['Residence_type'])
data['smoking_status'] = le.fit_transform(data['smoking_status'])

y = data['stroke'].values
x = data.drop('stroke', axis=1)
print(data.head(5))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=11, shuffle=True, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = knn.score(x_test, y_test)
print('Accuracy:', accuracy)
with open('model2.pkl', 'wb') as f:
    pickle.dump(knn, f)