import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm

def load_all_subjects(data_dir='./WESAD'):
    subject_dirs = [s for s in os.listdir(data_dir) if s.startswith('S')]
    subjects = []
    for sd in subject_dirs:
        path = os.path.join(data_dir, sd, f"{sd}.pkl")
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        subjects.append((sd, data))
    return subjects

def extract_features(subject_data, window_size=700*60, step_size=700*30):

    subj_id, data = subject_data
    chest = data['signal']['chest']
    labels = data['label']
    features_list = []

    for start in range(0, len(labels) - window_size, step_size):
        end = start + window_size
        label_window = labels[start:end]

        if len(label_window) == 0 or np.all(label_window == 0):
            continue

        label = np.bincount(label_window).argmax()
        feats = {'subject': subj_id, 'label': label}

        for signal_name, signal_data in chest.items():
            try:
                signal_window = signal_data[start:end]

                if signal_window.ndim == 2:
                    for i in range(signal_window.shape[1]):
                        col_data = signal_window[:, i]
                        feats[f'{signal_name}_{i}_mean'] = np.mean(col_data)
                        feats[f'{signal_name}_{i}_std'] = np.std(col_data)
                        feats[f'{signal_name}_{i}_min'] = np.min(col_data)
                        feats[f'{signal_name}_{i}_max'] = np.max(col_data)
                else:
                    feats[f'{signal_name}_mean'] = np.mean(signal_window)
                    feats[f'{signal_name}_std'] = np.std(signal_window)
                    feats[f'{signal_name}_min'] = np.min(signal_window)
                    feats[f'{signal_name}_max'] = np.max(signal_window)

                # NeuroKit2 feature extraction
                if signal_name == 'ECG':
                    processed, info = nk.ecg_process(signal_window[:, 0], sampling_rate=700)
                    nk_feats = nk.ecg_intervalrelated(processed)
                    feats.update(nk_feats.iloc[0].to_dict())
                elif signal_name == 'EDA':
                    processed, info = nk.eda_process(signal_window[:, 0], sampling_rate=700)
                    nk_feats = nk.eda_intervalrelated(processed)
                    feats.update(nk_feats.iloc[0].to_dict())
                elif signal_name == 'RESP':
                    processed, info = nk.rsp_process(signal_window[:, 0], sampling_rate=700)
                    nk_feats = nk.rsp_intervalrelated(processed)
                    feats.update(nk_feats.iloc[0].to_dict())
                elif signal_name == 'EMG':
                    processed, info = nk.emg_process(signal_window[:, 0], sampling_rate=700)
                    nk_feats = nk.emg_intervalrelated(processed)
                    feats.update(nk_feats.iloc[0].to_dict())

            except Exception as e:
                print(f"[{subj_id}] {signal_name} error: {e}")
        features_list.append(feats)

    return pd.DataFrame(features_list)

# Load and extract
data_dir = r'C:\Users\hadja\Downloads\WESAD'
subjects = load_all_subjects(data_dir)
print(f"Total subjects loaded: {len(subjects)}")

df_all = pd.DataFrame()
for subj in tqdm(subjects):
    df = extract_features(subj)
    df_all = pd.concat([df_all, df], ignore_index=True)

# Clean and save
df_all = df_all.dropna(axis=1, thresh=len(df_all) * 0.9)
df_all = df_all.fillna(df_all.median(numeric_only=True))
csv_path = os.path.join(data_dir, "wesad_full_features.csv")
df_all.to_csv(csv_path, index=False)
print(f"✅ All features extracted and saved to {csv_path}")
