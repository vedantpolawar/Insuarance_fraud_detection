from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

FEATURE_NAMES = [
    'months_as_customer', 'policy_deductable', 'umbrella_limit',
    'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'injury_claim', 'property_claim', 'vehicle_claim', 'age',
    'total_claim_amount', 'policy_csl', 'insured_sex',
    'insured_education_level', 'insured_occupation',
    'insured_relationship', 'incident_type', 'collision_type',
    'incident_severity', 'authorities_contacted',
    'property_damage', 'police_report_available'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            'months_as_customer':       int(data['months_as_customer']),
            'age':                      int(data['age']),
            'policy_deductable':        int(data['policy_deductable']),
            'umbrella_limit':           int(data['umbrella_limit']),
            'capital-gains':            int(data['capital_gains']),
            'capital-loss':             int(data['capital_loss']),
            'incident_hour_of_the_day': int(data['incident_hour']),
            'number_of_vehicles_involved': int(data['num_vehicles']),
            'bodily_injuries':          int(data['bodily_injuries']),
            'witnesses':                int(data['witnesses']),
            'injury_claim':             int(data['injury_claim']),
            'property_claim':           int(data['property_claim']),
            'vehicle_claim':            int(data['vehicle_claim']),
            'total_claim_amount':       int(data['total_claim_amount']),
            'policy_csl':               data['policy_csl'],
            'insured_sex':              data['insured_sex'],
            'insured_education_level':  data['insured_education_level'],
            'insured_occupation':       data['insured_occupation'],
            'insured_relationship':     data['insured_relationship'],
            'incident_type':            data['incident_type'],
            'collision_type':           data['collision_type'],
            'incident_severity':        data['incident_severity'],
            'authorities_contacted':    data['authorities_contacted'],
            'property_damage':          data['property_damage'],
            'police_report_available':  data['police_report_available'],
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        fraud_prob = float(probability[1]) if prediction == 'Y' else float(probability[0])
        fraud_label = 'FRAUDULENT' if prediction == 'Y' else 'LEGITIMATE'
        fraud_score = float(probability[list(model.classes_).index('Y')]) * 100

        return jsonify({
            'success': True,
            'prediction': fraud_label,
            'fraud_score': round(fraud_score, 1),
            'is_fraud': prediction == 'Y',
            'confidence': round(max(probability) * 100, 1)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'Insurance Fraud Detector (Logistic Regression)'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)