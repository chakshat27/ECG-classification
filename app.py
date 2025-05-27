# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fft, fftfreq
# import pywt
# import joblib


# # Cleaning helper function
# def str_to_clean_float_list(s):
#     try:
#         items = s.strip('[]').split(',')
#         floats = []
#         for item in items:
#             item = item.strip()
#             if item and item not in ['-', '--']:
#                 try:
#                     floats.append(float(item))
#                 except ValueError:
#                     continue
#         return floats
#     except Exception:
#         return []

# def clean_uploaded_ecg(df, min_signal_length=100):
#     df['Signal'] = df['Signal'].apply(str_to_clean_float_list)
#     df = df[df['Signal'].apply(lambda x: isinstance(x, list) and len(x) >= min_signal_length)]
#     df.reset_index(drop=True, inplace=True)
#     return df


# # Load the trained model
# model_file = st.file_uploader("Upload Trained Model (.pkl)", type=["pkl"])
# if model_file is not None:
#     model = joblib.load(model_file)
# else:
#     st.warning("Please upload a trained model file (.pkl) to proceed.")
#     st.stop()
# # Ensure the model file is in the same directory

# # Convert string to list of floats
# # def str_to_float_list(s):
# #     return [float(item) for item in s.strip('[]').split(',') if item.strip() != '']


# # -------------------
# # Cleaner function
# # # -------------------
# # def str_to_clean_float_list(s):
# #     try:
# #         items = s.strip('[]').split(',')
# #         floats = []
# #         for item in items:
# #             item = item.strip()
# #             if item and not item in ['-', '--']:
# #                 try:
# #                     floats.append(float(item))
# #                 except ValueError:
# #                     continue
# #         return floats
# #     except Exception:
# #         return []

# # def clean_uploaded_ecg(df, min_signal_length=100):
# #     df['Signal'] = df['Signal'].apply(str_to_clean_float_list)
# #     df = df[df['Signal'].apply(lambda x: isinstance(x, list) and len(x) >= min_signal_length)]
# #     df.reset_index(drop=True, inplace=True)
# #     return df


# # Feature extraction functions
# def extract_fft_features(signal, sampling_rate=300):
#     N = len(signal)
#     T = 1.0 / sampling_rate
#     yf = fft(signal)
#     xf = fftfreq(N, T)[:N//2]
#     mag = 2.0 / N * np.abs(yf[:N//2])

#     mask = (xf >= 0.5) & (xf <= 40)
#     freqs = xf[mask]
#     mags = mag[mask]

#     features = {
#         'fft_mean_mag': np.mean(mags),
#         'fft_max_mag': np.max(mags),
#         'fft_freq_max_power': freqs[np.argmax(mags)],
#         'fft_power_band_energy': np.sum(mags**2),
#         'fft_low_freq_power_ratio': np.sum(mags[freqs < 5]) / np.sum(mags)
#     }
#     return features

# def extract_wavelet_features(signal, wavelet='db4', level=5):
#     coeffs = pywt.wavedec(signal, wavelet, level=level)
#     features = {}
#     for i, coeff in enumerate(coeffs):
#         features[f'dwt_coeff_{i}_mean'] = np.mean(coeff)
#         features[f'dwt_coeff_{i}_std'] = np.std(coeff)
#         features[f'dwt_coeff_{i}_energy'] = np.sum(np.square(coeff))
#     return features

# # Streamlit App
# st.title("ECG Signal Classification GUI")

# # File upload
# uploaded_file = st.file_uploader("Upload your ECG CSV file (must contain 'Signal' column):", type=["csv"])

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)

#         if 'Signal' not in df.columns:
#             st.error("Uploaded CSV must contain a 'Signal' column.")
#         else:
#             # Convert string to list of floats
#             df.Signal = df.Signal.apply(str_to_clean_float_list)

#             st.success("File uploaded and processed successfully.")

#             signal_index = st.number_input("Select Signal Index:", min_value=0, max_value=len(df)-1, step=1)

#             if st.button("Classify Signal"):

#                 signal = df.Signal[signal_index]

#                 # Plot signal
#                 st.subheader("ECG Signal Plot")
#                 time_vector = np.arange(len(signal)) / 300
#                 fig, ax = plt.subplots()
#                 ax.plot(time_vector, signal)
#                 ax.set_xlabel("Time (s)")
#                 ax.set_ylabel("Amplitude")
#                 ax.set_title(f"ECG Signal at Index {signal_index}")
#                 ax.grid(True)
#                 st.pyplot(fig)

#                 # Extract features
#                 fft_features = extract_fft_features(signal)
#                 dwt_features = extract_wavelet_features(signal)
#                 all_features = {**fft_features, **dwt_features}
#                 features_df = pd.DataFrame([all_features])

#                 # Predict class
#                 prediction = model.predict(features_df)[0]
#                 st.success(f"Predicted Class: **{prediction}**")

#                 if st.checkbox("Show Extracted Features"):
#                     st.dataframe(features_df.T.rename(columns={0: 'Value'}))

#     except Exception as e:
#         st.error(f"Error processing the file: {e}")
# else:
#     st.info("Please upload a CSV file to begin.")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import pywt
import joblib

# --- Helper Functions ---

def str_to_clean_float_list(s):
    try:
        items = s.strip('[]').split(',')
        floats = []
        for item in items:
            item = item.strip()
            if item and item not in ['-', '--']:
                try:
                    floats.append(float(item))
                except ValueError:
                    continue
        return floats
    except Exception:
        return []

def clean_uploaded_ecg(df, min_signal_length=100):
    df['Signal'] = df['Signal'].apply(str_to_clean_float_list)
    df = df[df['Signal'].apply(lambda x: isinstance(x, list) and len(x) >= min_signal_length)]
    df.reset_index(drop=True, inplace=True)
    return df

# --- Feature Extraction ---

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

def extract_features(signal, fs=300):
    signal = np.array(signal)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    adaptive_height = mean_val + 1.2 * std_val

    peaks, _ = find_peaks(signal, height=adaptive_height, distance=int(0.3 * fs))
    rr_intervals = np.diff(peaks) / fs
    bpm = 60 / rr_intervals if len(rr_intervals) > 1 else []

    features = {
        "mean_bpm": np.mean(bpm) if len(bpm) > 0 else 0,
        "std_bpm": np.std(bpm) if len(bpm) > 0 else 0,
        "mean_rr": np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
        "std_rr": np.std(rr_intervals) if len(rr_intervals) > 0 else 0,
        "median_rr": np.median(rr_intervals) if len(rr_intervals) > 0 else 0,
        "iqr_rr": np.percentile(rr_intervals, 75) - np.percentile(rr_intervals, 25) if len(rr_intervals) > 0 else 0,
        "rmssd": np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) > 1 else 0,
        "nni_50": np.sum(np.abs(np.diff(rr_intervals)) > 0.05),
        "nni_20": np.sum(np.abs(np.diff(rr_intervals)) > 0.02),
        "pnn50": np.mean(np.abs(np.diff(rr_intervals)) > 0.05) if len(rr_intervals) > 1 else 0,
        "peak_count": len(peaks),
        "signal_std": np.std(signal),
        "signal_mean": np.mean(signal),
        "signal_max": np.max(signal),
        "signal_min": np.min(signal)
    }
    return features

# --- Streamlit App ---

st.title("ECG Signal Classification GUI")

# Upload model
model_file = st.file_uploader("Upload Trained Model (.pkl)", type=["pkl"])
if model_file is not None:
    loaded_obj = joblib.load(model_file)
    if isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
        model, feature_names = loaded_obj
    else:
        model = loaded_obj
        feature_names = None
else:
    st.warning("Please upload a trained model file (.pkl) to proceed.")
    st.stop()

# Upload ECG CSV
uploaded_file = st.file_uploader("Upload your ECG CSV file (must contain 'Signal' column):", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'Signal' not in df.columns:
            st.error("Uploaded CSV must contain a 'Signal' column.")
        else:
            df = clean_uploaded_ecg(df)

            st.success("File uploaded and processed successfully.")

            signal_index = st.number_input("Select Signal Index:", min_value=0, max_value=len(df)-1, step=1)

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

                # Extract all features
                stats = extract_features(signal, fs=300)
                fft_feats = extract_fft_features(signal)
                dwt_feats = extract_wavelet_features(signal)
                all_features = {**stats, **fft_feats, **dwt_feats}

                features_df = pd.DataFrame([all_features])
                if feature_names:
                    features_df = features_df.reindex(columns=feature_names, fill_value=0)

                # Predict
                prediction = model.predict(features_df)[0]
                st.success(f"Predicted Class: **{prediction}**")

                if st.checkbox("Show Extracted Features"):
                    st.dataframe(features_df.T.rename(columns={0: 'Value'}))

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
