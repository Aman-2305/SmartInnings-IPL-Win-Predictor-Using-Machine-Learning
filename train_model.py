import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import os

df = pd.read_csv("data/ball_by_ball_ipl.csv")

df = df[df['Innings'] == 2].copy()

df = df.dropna(subset=[
    'Bat First', 'Bat Second', 'Venue',
    'Target Score', 'Innings Runs', 'Balls Remaining',
    'Runs to Get', 'Innings Wickets', 'Chased Successfully'
])

df['runs_left'] = df['Runs to Get']
df['balls_left'] = df['Balls Remaining']
df['wickets_left'] = 10 - df['Innings Wickets']
df['target'] = df['Target Score']
df['crr'] = df['Innings Runs'] / ((120 - df['balls_left']) / 6)
df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df[df['balls_left'] > 0]

features = ['Bat First', 'Bat Second', 'Venue', 'runs_left', 'balls_left',
            'wickets_left', 'target', 'crr', 'rrr']
X = df[features]
y = df['Chased Successfully'] 

label_encoders = {}
for col in ['Bat First', 'Bat Second', 'Venue']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(label_encoders, "models/encoders.pkl")











# correlation tests (dont include)

print("\n Feature Correlation with Target (Chased Successfully):")
correlations = X.copy()
correlations['target'] = y
corr_matrix = correlations.corr()
target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
print(target_corr)

print("\n XGBoost Feature Importances:")
importances = model.feature_importances_
for feature, score in zip(X.columns, importances):
    print(f"{feature:20s}: {score:.4f}")
    
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances, color='skyblue')
plt.title("XGBoost Feature Importances")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("xgboost_feature_importance.png")
plt.show()
