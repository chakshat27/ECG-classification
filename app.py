import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pywt
import joblib  # use joblib or pickle depending on your saved model

# Load the CSV and preprocess
df = pd.read_csv('Updated_reference.csv')  # Adjust path as needed

# Convert string to list of floats
def str_to_float_list(s):
    return [float(item) for item in s.strip('[]').split(',') if item.strip() != '']

df.Signal = df.Signal.apply(str_to_float_list)

# Feature extraction functions
def extract_fft_features(signal, sampling_rate=300):
    N = len(signal)
    T = 1.0 / sampling_rate
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]
    mag = 2.0 / N * np.abs(yf[:N//2])

    mask = (xf >= 0.5) & (xf <= 40)
    freqs = xf[mask]
    mags = mag[mask]

    features = {
        'fft_mean_mag': np.mean(mags),
        'fft_max_mag': np.max(mags),
        'fft_freq_max_power': freqs[np.argmax(mags)],
        'fft_power_band_energy': np.sum(mags**2),
        'fft_low_freq_power_ratio': np.sum(mags[freqs < 5]) / np.sum(mags)
    }
    return features

def extract_wavelet_features(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'dwt_coeff_{i}_mean'] = np.mean(coeff)
        features[f'dwt_coeff_{i}_std'] = np.std(coeff)
        features[f'dwt_coeff_{i}_energy'] = np.sum(np.square(coeff))
    return features

# Load your pre-trained model
# model = joblib.load('model.pkl')  # Uncomment when model is available

# Fake classifier (mock)
def mock_classifier(features_df):
    classes = ['Normal', 'AFib', 'Noise', 'Other']
    return np.random.choice(classes)

# Streamlit App
st.title("ECG Signal Classification GUI")

signal_index = st.number_input("Enter Signal Index (0 to {}):".format(len(df)-1), min_value=0, max_value=len(df)-1, step=1)

if st.button("Classify Signal"):

    signal = df.Signal[signal_index]

    # Plot signal
    st.subheader("ECG Signal Plot")
    time_vector = np.arange(len(signal)) / 300
    fig, ax = plt.subplots()
    ax.plot(time_vector, signal)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"ECG Signal at Index {signal_index}")
    ax.grid(True)
    st.pyplot(fig)

    # Extract features
    fft_features = extract_fft_features(signal)
    dwt_features = extract_wavelet_features(signal)
    all_features = {**fft_features, **dwt_features}
    features_df = pd.DataFrame([all_features])

    # Predict class
    # prediction = model.predict(features_df)[0]  # Use this line with real model
    prediction = mock_classifier(features_df)  # Mock output

    st.success(f"Predicted Class: **{prediction}**")

    # Optional: Display extracted features
    if st.checkbox("Show Extracted Features"):
        st.dataframe(features_df.T.rename(columns={0: 'Value'}))
