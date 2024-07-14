from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the models, scaler, and label encoder classes
models = {
    "Logistic Regression": joblib.load('Logistic Regression.pkl'),
    "Decision Tree Classifier": joblib.load('Decision Tree Classifier.pkl'),
    "Random Forest Classifier": joblib.load('Random Forest Classifier.pkl'),
    "XGBoost": joblib.load('XGBoost.pkl'),
    "Multi-Layer Perceptron": joblib.load('Multi-Layer Perceptron.pkl')
}
scaler = joblib.load('scaler.pkl')
label_encoder_classes = joblib.load('classes.pkl')
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        data = pd.read_csv(file)

        if 'Congestion_Type' in data.columns:
            data = data.drop('Congestion_Type', axis=1)

        categorical_cols = ['cell_name', '4G_rat', 'ran_vendor']
        for col in categorical_cols:
            data[col] = data[col].fillna('Unknown')  # Fill NaNs with 'Unknown'
            if 'Unknown' not in label_encoder.classes_:
                label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')
            data[col] = data[col].apply(lambda x: x if x in label_encoder.classes_ else 'Unknown')
            data[col] = label_encoder.transform(data[col])

        try:
            data_scaled = scaler.transform(data)
        except ValueError as e:
            print(f"Error scaling data: {e}")
            return "Data scaling failed due to unseen values."

        predictions = {}
        for name, model in models.items():
            preds = model.predict(data_scaled)
            try:
                decoded_preds = label_encoder.inverse_transform(preds)
                predictions[name] = decoded_preds.tolist()
            except ValueError as e:
                print(f"Error decoding predictions for {name}: {e}")
                predictions[name] = ["Prediction decoding failed."]

        return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
