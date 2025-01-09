from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np


data = pd.read_csv('video_frame_data.csv')
# print(data.head())

X= data[['frame_index','pixel_diff','entropy','pixel_intensity_var','edge','absolute_diff']]
y= data[['frame_type']]

# print(X.head())

# Assuming 'X' contains feature columns and 'y' is the target column
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)



y_pred = rf_model.predict(X_test)

# Performance Metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
