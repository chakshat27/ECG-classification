
---

#  ECG Signal Classification using Machine Learning

An end-to-end **ECG (Electrocardiogram) signal classification system** that analyzes raw waveform data, extracts meaningful features, and predicts cardiac rhythm using a trained machine learning model.

This project combines **signal processing + feature engineering + ML + deployment (Streamlit UI)**.

---

## 🚀 Features

* 📥 Upload ECG signals in CSV format
* 🧹 Automatic signal cleaning & preprocessing
* 📊 Feature extraction:

  * Time-domain (RR intervals, BPM)
  * Frequency-domain (FFT)
  * Wavelet-based (DWT)
* 🧠 Machine Learning classification (Random Forest)
* 📈 ECG waveform visualization
* ⚡ Real-time prediction via Streamlit UI

---

## 🧠 Problem Statement

Classify ECG signals into:

| Label | Meaning             |
| ----- | ------------------- |
| **N** | Normal Sinus Rhythm |
| **A** | Atrial Fibrillation |
| **O** | Other Rhythm        |
| **~** | Noisy Signal        |

---


---

## ⚙️ Pipeline Overview

### 1. Data Preprocessing

* Convert signal strings → float arrays
* Remove invalid / short signals
* Normalize and clean noisy values

---

### 2. Signal Processing

#### 📡 FFT (Frequency Analysis)

* Extract dominant frequencies
* Compute power spectrum features

#### 🌊 DWT (Wavelet Transform)

* Multi-resolution signal decomposition
* Extract:

  * Mean
  * Standard deviation
  * Energy
  * Entropy

#### ❤️ R-Peak Detection

* Adaptive thresholding:

  * `mean + 1.2 * std`
* Extract:

  * RR intervals
  * BPM
  * Heart rate variability metrics

---

### 3. Feature Engineering

Extracted features include:

* Mean BPM, Std BPM
* RR interval stats (mean, std, median, IQR)
* RMSSD, pNN50, NN50
* Signal statistics (mean, max, min, std)
* FFT features
* Wavelet features

---

### 4. Model Training

* Model: **Random Forest Classifier**
* Train-test split: 80/20
* Saved using `joblib`

```python
joblib.dump((clf, feature_names), 'ecg_classifier_model.pkl')
```

---

## 🖥️ Streamlit Application

### Inputs:

* Upload trained `.pkl` model
* Upload ECG CSV file (`Signal` column required)
* Select signal index

### Outputs:

* 📈 ECG waveform plot
* ❤️ Predicted heart rhythm

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/ecg-classification.git
cd ecg-classification
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas numpy matplotlib scipy pywavelets scikit-learn joblib
```

---

### 3. Run the App

```bash
streamlit run app.py
```

---

## 📊 Sample Workflow

1. Upload trained model (`.pkl`)
2. Upload ECG dataset (`.csv`)
3. Select signal index
4. Click **"Classify Signal"**
5. View:

   * Signal plot
   * Predicted class

---

## 📌 Key Highlights

* Combines **signal processing + ML**
* Handles **real-world noisy ECG data**
* Uses **adaptive peak detection**
* Fully interactive UI
* Modular & extensible pipeline

---

## 🧪 Future Improvements

* Deep Learning (CNN / LSTM for ECG)
* Real-time ECG streaming support
* Advanced denoising (Kalman filter)
* Multi-lead ECG support
* Deployment on cloud (AWS / GCP / Streamlit Cloud)

---


---

## 📜 License

This project is open-source and available under the MIT License.

---
