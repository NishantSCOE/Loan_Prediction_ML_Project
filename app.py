from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os, warnings, logging

warnings.filterwarnings('ignore')
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)

models = {}
label_encoders = {}
feature_columns = []
model_accuracies = {}

def train_models():
    global models, label_encoders, feature_columns, model_accuracies

    csv_path = os.path.join(os.path.dirname(__file__), "loan_approval_dataset.csv")
    if not os.path.exists(csv_path):
        print("⚠  loan_approval_dataset.csv not found – models not trained.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    for col in ['loan_id', 'Loan_ID', 'id']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    target = 'loan_status'
    X = df.drop(target, axis=1)
    y = df[target]
    feature_columns[:] = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['lr'] = lr
    model_accuracies['lr'] = round(accuracy_score(y_test, lr.predict(X_test)) * 100, 2)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    models['dt'] = dt
    model_accuracies['dt'] = round(accuracy_score(y_test, dt.predict(X_test)) * 100, 2)

    print(f"✅ LR: {model_accuracies['lr']}%  |  DT: {model_accuracies['dt']}%")
    print(f"🌐 http://127.0.0.1:5000  (Press CTRL+C to quit)")

def safe_encode(col, val):
    if col not in label_encoders:
        return int(val)
    classes = label_encoders[col].classes_
    for c in classes:
        if str(c).strip().lower() == str(val).strip().lower():
            return int(label_encoders[col].transform([c])[0])
    return 0

def decode_status(encoded_val):
    if 'loan_status' in label_encoders:
        return label_encoders['loan_status'].inverse_transform([encoded_val])[0].strip()
    return 'Approved' if encoded_val == 0 else 'Rejected'

@app.route('/')
def index():
    return render_template('index.html', model_accuracies=model_accuracies, ready=bool(models))

@app.route('/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({'error': 'Models not trained. Please add loan_approval_dataset.csv'}), 400

    data = request.get_json()
    try:
        row = {
            'no_of_dependents'        : int(data['no_of_dependents']),
            'education'               : safe_encode('education', data['education']),
            'self_employed'           : safe_encode('self_employed', data['self_employed']),
            'income_annum'            : float(data['income_annum']),
            'loan_amount'             : float(data['loan_amount']),
            'loan_term'               : int(data['loan_term']),
            'cibil_score'             : int(data['cibil_score']),
            'residential_assets_value': float(data['residential_assets_value']),
            'commercial_assets_value' : float(data['commercial_assets_value']),
            'luxury_assets_value'     : float(data['luxury_assets_value']),
            'bank_asset_value'        : float(data['bank_asset_value']),
        }

        sample = pd.DataFrame([row])[feature_columns]

        results = {}
        for key, model in models.items():
            pred = model.predict(sample)[0]
            proba = model.predict_proba(sample)[0]
            status = decode_status(pred)
            results[key] = {
                'status'    : status,
                'confidence': round(float(max(proba)) * 100, 1),
                'approved'  : 'approved' in status.lower()
            }

        return jsonify({'success': True, 'results': results, 'accuracies': model_accuracies})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    return jsonify({'ready': bool(models), 'accuracies': model_accuracies})

train_models()

if __name__ == '__main__':
    app.run(debug=False, port=5000)