from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained stroke prediction model
model = pickle.load(open('model2.pkl','rb'))

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict_brain_stroke():
    age = float(request.form.get('age'))
    bmi = float(request.form.get('bmi'))
    glucose_level = float(request.form.get('avg_glucose_level'))
    gender = int(request.form.get('gender'))
    heart_disease = int(request.form.get('heart_disease'))
    hypertension = int(request.form.get('hypertension'))
    ever_married = int(request.form.get('ever_married'))
    residence_type = int(request.form.get('Residence_type'))
    work_type = int(request.form.get('work_type'))
    smoking_status = int(request.form.get('smoking_status'))
    
    if (heart_disease == 'Yes'):
        heart_disease = 1;
    else:
        heart_disease == 0;
        
    if (hypertension == 'Yes'):
        hypertension= 1;
    else:
        hypertension == 0;
    result = model.predict(np.array([age, bmi, glucose_level, gender, heart_disease, hypertension, ever_married, residence_type, work_type, smoking_status]).reshape(1, -1))
    if (result[0] == 1):
        result = 'Risk Of Stroke :Low'
        
    else:
        result = 'Risk Of Stroke :High'
    return render_template('index.html',result = result)

if __name__ == '__main__':
    app.run(debug=True)
