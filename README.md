# LoanLens – Loan Approval Predictor (Flask)

## Project Structure
```
loan_app/
├── app.py                      ← Flask backend + ML training
├── requirements.txt
├── loan_approval_dataset.csv   ← ⚠ YOU MUST ADD THIS FILE
└── templates/
    └── index.html              ← Frontend UI
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
Copy `loan_approval_dataset.csv` into the `loan_app/` folder  
(same directory as `app.py`).

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## Features
- Trains **Logistic Regression** and **Decision Tree** models automatically on startup
- Shows live model accuracy in the header
- **Quick Presets** to test Strong / Weak applicant scenarios
- Confidence bars for each model's prediction
- Fully responsive UI

## Notes
- The models are re-trained every time you start the app (uses your CSV directly)
- No manual training step needed — just run `python app.py`
