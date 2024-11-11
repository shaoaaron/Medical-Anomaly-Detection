import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve



record = wfdb.rdrecord('/Users/aaronshao/Documents/Personal-Projects/Medical-Anomaly-Detection/af-termination-challenge-database-1.0.0/test-set-a/a01')  

def plot_ecg(record_name):
    record = wfdb.rdrecord(record_name)
    plt.plot(record.p_signal[:, 0])  
    plt.title(f'ECG Signal for {record_name}')
    plt.xlabel('Time (samples)')
    plt.ylabel('ECG Amplitude')
    plt.show()

def check_record_info(record_name):
    record = wfdb.rdrecord(record_name)
    print(f'Record: {record_name}')
    print(f'Number of signals: {record.n_sig}')
    print(f'Sample rate (Hz): {record.fs}')
    print(f'Signal length: {len(record.p_signal)} samples')
    print()

def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=128, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

record_name = '/Users/aaronshao/Documents/Personal-Projects/Medical-Anomaly-Detection/af-termination-challenge-database-1.0.0/test-set-a/a01'
record = wfdb.rdrecord(record_name)

lowcut = 0.5  
highcut = 50.0  
fs = 128  

filtered_signal_1 = bandpass_filter(record.p_signal[:, 0], lowcut, highcut, fs)
filtered_signal_2 = bandpass_filter(record.p_signal[:, 1], lowcut, highcut, fs)

def interpolate_missing_data(signal):
    nans = np.isnan(signal)
    signal[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), signal[~nans])
    return signal

filtered_signal_1_cleaned = interpolate_missing_data(filtered_signal_1)
filtered_signal_2_cleaned = interpolate_missing_data(filtered_signal_2)

# Min-max normalization
def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val)

# Signal normalization
normalized_signal_1 = normalize_signal(filtered_signal_1_cleaned)
normalized_signal_2 = normalize_signal(filtered_signal_2_cleaned)

# Extract statistics
def extract_statistical_features(signal):
    mean = np.mean(signal)        
    std = np.std(signal)          
    signal_skew = skew(signal)    
    signal_kurtosis = kurtosis(signal)  
    
    return [mean, std, signal_skew, signal_kurtosis]

all_features = []

for record_num in range(1, 31):  
    record_name = f"test-set-a/a{str(record_num).zfill(2)}" 
    try:
        record = wfdb.rdrecord(f'/Users/aaronshao/Documents/Personal-Projects/Medical-Anomaly-Detection/af-termination-challenge-database-1.0.0/{record_name}')
        
        filtered_signal_1 = bandpass_filter(record.p_signal[:, 0])
        filtered_signal_2 = bandpass_filter(record.p_signal[:, 1])
        
        features_signal_1 = extract_statistical_features(filtered_signal_1)
        features_signal_2 = extract_statistical_features(filtered_signal_2)
        
        combined_features = [record_name] + features_signal_1 + features_signal_2
        all_features.append(combined_features)
    
    except Exception as e:
        print(f"Error processing record {record_name}: {e}")

columns = [
    "record", 
    "mean_signal_1", "std_signal_1", "skew_signal_1", "kurtosis_signal_1",
    "mean_signal_2", "std_signal_2", "skew_signal_2", "kurtosis_signal_2"
]
df_features = pd.DataFrame(all_features, columns=columns)

df = pd.read_csv("extracted_features_test_set_a.csv")

zscore_threshold = 0.5 
df['label'] = np.where(abs(zscore(df['mean_signal_1'])) > zscore_threshold, 'Anomalous', 'Normal')

# print("Initial label distribution:\n", df['label'].value_counts())

X = df.drop('label', axis=1) 
y = df['label']  

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

train_class_counts = pd.Series(y_train).value_counts()
# print("Training set label distribution:\n", train_class_counts)

smote = SMOTE(sampling_strategy=0.75, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_resampled, y_resampled)

y_prob = model.predict_proba(X_test)[:, 1]  
y_proba = model.predict_proba(X_test)[:, 1]  

precision, recall, thresholds = precision_recall_curve(y_test, y_proba, pos_label=1)
optimal_threshold = thresholds[np.argmax(precision + recall)]  

y_pred_adjusted = (y_prob >= optimal_threshold).astype(int)

f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

def detect_anomaly(input_data, model, scaler, threshold=optimal_threshold, feature_names=None):
    """
    Detects if the input data is anomalous based on the trained model.
    
    Parameters:
        input_data (array-like): New data input (single sample) to classify.
        model (object): The trained classification model.
        scaler (object): The fitted scaler to transform input data.
        threshold (float): Probability threshold for classifying as "Anomalous."
        feature_names (list of str): Column names used during model training.
        
    Returns:
        str: "Anomalous" or "Normal"
    """
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    input_scaled = scaler.transform(input_df)
    
    proba_anomalous = model.predict_proba(input_scaled)[0][1]
    
    return "Anomalous" if proba_anomalous >= threshold else "Normal"

feature_names = X.columns  
new_data = [0.5, 0.1, -0.2, 0.3, -1.2, 0.7, -0.6, 1.3]  # Replace with actual values that the user wants to test 
result = detect_anomaly(new_data, model, scaler, feature_names=feature_names)
print("Detection Result:", result)

record_path = '/Users/aaronshao/Documents/Personal-Projects/Medical-Anomaly-Detection/af-termination-challenge-database-1.0.0/test-set-a/a01'












# df_zscore = df.apply(zscore)

# anomalies = df_zscore[abs(df_zscore) > 3]
# print(f"Anomalies detected at these indices: {anomalies.dropna().index}")

# for i in range(1, 31):  
#     record_name = f'/Users/aaronshao/Documents/Personal-Projects/Medical-Anomaly-Detection/af-termination-challenge-database-1.0.0/test-set-a/a{str(i).zfill(2)}' 
#     plot_ecg(record_name)

# for i in range(1, 31): 
#     record_name = f'/Users/aaronshao/Documents/Personal-Projects/Medical-Anomaly-Detection/af-termination-challenge-database-1.0.0/test-set-a/a{str(i).zfill(2)}'
#     check_record_info(record_name)

# filtered_signal = bandpass_filter(record.p_signal[:, 0])

# Plot original vs. filtered signal
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(record.p_signal[:, 0])
# plt.title('Original ECG Signal')
# plt.subplot(2, 1, 2)
# plt.plot(filtered_signal)
# plt.title('Filtered ECG Signal')
# plt.show()

# plt.plot(record.p_signal[:, 0], label='Signal 1')  # First signal (channel)
# plt.plot(record.p_signal[:, 1], label='Signal 2')  # Second signal (channel)
# plt.legend()
# plt.show()

# df_features.to_csv("extracted_features_test_set_a.csv", index=False)
# print("Feature extraction completed and saved to extracted_features_test_set_a.csv")

# print(df.head())

# print(df.isnull().sum())

# print(df.describe())

# if len(train_class_counts) > 1:
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# else:
#     print("Only one class present in y_train after splitting; skipping SMOTE.")
#     X_resampled, y_resampled = X_train, y_train 

# if len(np.unique(y_resampled)) > 1:
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_resampled, y_resampled)

#     # Predict on the test set
#     y_pred = model.predict(X_test)

#     # Evaluate the model
#     print("Model Performance on Test Set:")
#     print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))
#     print("Accuracy:", accuracy_score(y_test, y_pred))
# else:
#     print("Unable to train model due to lack of class diversity in training data.")




