import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
dataset = pd.read_csv('train_upd.csv')

# Data preprocessing
X = dataset.drop('Congestion_Type', axis=1)
y = dataset['Congestion_Type']

# Identify categorical columns
categorical_cols = ['cell_name', '4G_rat', 'ran_vendor']  # Add other categorical columns as necessary

# Encode categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encoding the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection
# Univariate selection using mutual_info_classif
bestfeatures = SelectKBest(score_func=mutual_info_classif, k='all')
fit = bestfeatures.fit(X_train, y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(dataset.columns[:-1])
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']

# Recursive Feature Elimination
model_lr = LogisticRegression(max_iter=10000)
rfe = RFE(estimator=model_lr, n_features_to_select=10)
fit = rfe.fit(X_train, y_train)

# Feature importance using decision tree
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)
importances = model_tree.feature_importances_
indices = np.argsort(importances)[::-1]

# Modelling
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Multi-Layer Perceptron": MLPClassifier(max_iter=1000)
}

model_accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    joblib.dump(model, f'{name}.pkl')

# Save the label encoder and scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder.classes_, 'classes.pkl')

# Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x='Specs', y='Score', data=featureScores.nlargest(10, 'Score'))
plt.xticks(rotation=90)
plt.title('Top 10 Features by Univariate Selection')
plt.tight_layout()
plt.savefig('univariate_selection.png')

plt.figure(figsize=(10, 6))
plt.title("Feature Importances using Decision Tree")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig('feature_importance_decision_tree.png')

print(model_accuracies)
