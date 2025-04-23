import xgboost as xgb
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS
import joblib
import pickle

app = Flask(__name__)
CORS(app)

# Load dataset and model
CLEAN_DATA_PATH = "data/clean_dataset.tsv"
MODEL_PATH = "model/xgboost_model.json"
SYMPTOM_DESC_PATH = "data/symptom_Description.csv"
SYMPTOM_PRECAUTION_PATH = "data/symptom_precaution.csv"

df = pd.read_csv(CLEAN_DATA_PATH, sep="\t")
symptom_columns = df.columns[:-1]

# 加载 scaler 和模型
heartScaler = joblib.load('model/HeartDiseaseLRScaler.pkl')  # 使用 joblib 加载 scaler
logistic_model = joblib.load('model/HeartDisease_Logistic.pkl')  # 使用 joblib 加载模型
diabetesScaler = joblib.load('model/DiabetesScaler.pkl')
diabetesPCA    = joblib.load('model/DiabetesPCA.pkl')
diabetesModel  = joblib.load('model/DiabetesModel.pkl')
diabetesEncoders = joblib.load('model/DiabetesEncoders.pkl')
xg_model = joblib.load('model/covid19_xgboost.pkl')
liver_model = joblib.load('model/LiverDisease_LR.pkl')
liver_encoders = joblib.load('model/LiverEncoders.pkl')

# 更新后的编码映射
heart_label_encodings = {
    'Sex': {'Female': 0, 'Male': 1},
    'ChestPainType': {
        'Typical Angina': 1,
        'Atypical Angina': 2,
        'Non-Anginal Pain': 3,
        'Asymptomatic': 4
    },
    'FastingBloodSugarOver120': {'False': 0, 'True': 1},
    'RestingECGResult': {
        'Normal': 0,
        'ST-T Abnormality': 1,
        'Left Ventricular Hypertrophy': 2
    },
    'ExerciseInducedAngina': {'No': 0, 'Yes': 1},
    'ST_Slope': {
        'Upsloping': 1,
        'Flat': 2,
        'Downsloping': 3
    },
    'Thallium': {
        'Normal': 3,
        'Fixed Defect': 6,
        'Reversible Defect': 7
    }
}

def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

# Preprocess input symptoms
def prepare_input(symptoms):
    symptoms_array = np.zeros((1, len(symptom_columns)))
    for symptom in symptoms:
        if symptom in symptom_columns:
            symptom_idx = df.columns.get_loc(symptom)
            symptoms_array[0, symptom_idx] = 1
    return symptoms_array

# Predict disease based on symptoms
def predict_disease(symptoms):
    X = prepare_input(symptoms)
    disease_idx = model.predict(X)[0]
    disease_prob = model.predict_proba(X)[0, disease_idx]
    disease_name = df.iloc[:, -1].astype('category').cat.categories[disease_idx]
    
    return disease_name, disease_prob

# Get disease description
def get_disease_description(disease_name):
    desc_df = pd.read_csv(SYMPTOM_DESC_PATH)
    desc = desc_df[desc_df["Disease"] == disease_name]["Description"]
    return desc.values[0] if not desc.empty else "No description available."

# Get disease precautions
def get_disease_precautions(disease_name):
    prec_df = pd.read_csv(SYMPTOM_PRECAUTION_PATH)
    prec = prec_df[prec_df["Disease"] == disease_name].filter(like="Precaution").values
    return prec[0].tolist() if len(prec) > 0 else ["No precautions available."]

# Flask API Endpoint
@app.route("/predict/disease", methods=["POST"])
def predict():
    data = request.json
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    disease, probability = predict_disease(symptoms)
    description = get_disease_description(disease)
    precautions = get_disease_precautions(disease)

    return jsonify({
        "disease": disease,
        "probability": float(probability),
        "description": description,
        "precautions": precautions
    })

# 处理传入数据
@app.route("/predict/heart", methods=["POST"])
def predict_heart_disease():
    try:
        data = request.get_json()

        features = [
            data["Age"],
            heart_label_encodings["Sex"][data["Sex"]],
            heart_label_encodings["ChestPainType"][data["ChestPainType"]],
            data["RestingBloodPressure"],
            data["SerumCholesterol"],
            heart_label_encodings["FastingBloodSugarOver120"][data["FastingBloodSugarOver120"]],
            heart_label_encodings["RestingECGResult"][data["RestingECGResult"]],
            data["MaxHeartRate"],
            heart_label_encodings["ExerciseInducedAngina"][data["ExerciseInducedAngina"]],
            data["ST_Depression"],
            heart_label_encodings["ST_Slope"][data["ST_Slope"]],
            data["NumberOfMajorVessels"],
            heart_label_encodings["Thallium"][data["Thallium"]]
        ]

        input_data = np.array([features])
        
        # Apply the scaler to scale the input data
        input_scaled = heartScaler.transform(input_data)  # Use the loaded scaler

        # Model prediction
        prediction = logistic_model.predict(input_scaled)
        # probability = logistic_model.predict_proba(input_scaled)[0][1]

        return jsonify({
            "has_heart_disease": bool(prediction),
            # "probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try: 
        data = request.get_json()

        # 编码字段
        gender = diabetesEncoders['gender'].transform([data['gender']])[0]
        smoking = diabetesEncoders['smoking_history'].transform([data['smoking_history']])[0]

        values = [[
            gender,
            data['age'],
            data['hypertension'],
            data['heart_disease'],
            smoking,
            data['bmi'],
            data['HbA1c_level'],
            data['blood_glucose_level']
        ]]

        columns = [
            'gender', 'age', 'hypertension', 'heart_disease',
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
        ]

        X_df = pd.DataFrame(values, columns=columns)
        X_scaled = diabetesScaler.transform(X_df)
        X_pca = diabetesPCA.transform(X_scaled)

        pred = diabetesModel.predict(X_pca)[0]
        prob = diabetesModel.predict_proba(X_pca)[0].max()

        return jsonify({
            'diabetes': int(pred),
            'probability': float(prob)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/covid', methods=['POST'])
def predict_covid():
    try:
        # 从前端获取数据
        data = request.get_json()

        input_data = {
            'Cough_symptoms': data['Cough_symptoms'], 
            'Fever': data['Fever'],  
            'Sore_throat': data['Sore_throat'],  
            'Shortness_of_breath': data['Shortness_of_breath'], 
            'Headache': data['Headache'], 
            'Age_60_above': data['Age_60_above'],
            'Sex': data['Sex'],  
            'Known_contact': data['Known_contact']
        }

        # 标准化输入数据中的布尔类型（True/False）值
        for col in ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']:
            if isinstance(input_data[col], str):
                # Replace 'TRUE' and 'FALSE' with 'True' and 'False'
                input_data[col] = input_data[col].strip().upper()  # Convert to uppercase for consistency
                input_data[col] = 'True' if input_data[col] == 'TRUE' else 'False'

        # 将输入数据转化为DataFrame
        X_df = pd.DataFrame([input_data])

        # 对每一列加载对应的LabelEncoder
        for col in input_data:
            # print(f"Loading LabelEncoder for column: {col}")
            try:
                le = joblib.load(f'model/covid/LabelEncoder_{col}.pkl')  # 从指定路径加载对应列的LabelEncoder
                # print(f"LabelEncoder for column {col} loaded successfully.")
                # # 检查当前列是否包含LabelEncoder适用的值
                # print(f"Available classes for {col}: {le.classes_}")
                X_df[col] = le.transform(X_df[col])  # 使用LabelEncoder进行转换
                # print(f"Encoded data for {col}: {X_df[col]}")
            except Exception as e:
                print(f"Error occurred while processing column {col}: {e}")
                raise e

        # print("Processed data after encoding:", X_df)

        # 进行预测
        covid_prediction = xg_model.predict(X_df)[0]
        covid_probability = xg_model.predict_proba(X_df)[0].max()

        return jsonify({
            'Corona': int(covid_prediction),  
            'probability': float(covid_probability) 
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/liver', methods=['POST'])
def predict_liver():
    try:
        data = request.get_json()

        columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
                   'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',  
                   'Albumin', 'Albumin_and_Globulin_Ratio']

        encoded_data = {}
        for col in columns:
            if col not in data:
                raise ValueError(f"Missing required field: {col}")
            
            if col == 'Gender':
                encoded_data[col] = liver_encoders['gender_encoder'].transform([data[col]])[0]
            else:
                encoded_data[col] = data[col]

        # print(f"Encoded data: {encoded_data}")
    
        df = pd.DataFrame([encoded_data])
        pred = liver_model.predict(df)[0]
        prob = liver_model.predict_proba(df)[0].max()

        print("the result is: ",pred)

        return jsonify({
            'liver_disease': int(pred),
            'probability': float(prob)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
# Start Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
