import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
# form sklearn.metrics import classification_report
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
from imblearn.over_sampling import SMOTENC
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=11, shuffle=True, stratify=y)
smote_nc = SMOTENC(categorical_features=[0,2,3,4,5,6,9], random_state=0)
X_resampled, Y_resampled = smote_nc.fit_resample(x_train, y_train)
SMOTE_df=pd.DataFrame(np.c_[X_resampled,Y_resampled],columns=data.columns)
df_selected=SMOTE_df[["age","avg_glucose_level","bmi","work_type","smoking_status","ever_married",'stroke']]
X_resampled,Y_resampled=df_selected[["age","avg_glucose_level","bmi","work_type","smoking_status","ever_married"]],df_selected[['stroke']]
clf = LogisticRegression().fit(x_train, y_train)
lpr=clf.predict(x_test)

with open('model3.pkl', 'wb') as f:
    pickle.dump(clf, f)