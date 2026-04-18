#  ICU Patient Vitals Anomaly Detection

##  Problem Statement

Build a machine learning model that identifies abnormal patterns in **real-time vital sign monitoring data** from surgical ICU patients. Given time-series vital signs data, predict an **anomaly_score (0 → 1)** for each timestamp, where:
- **0** = Normal vital sign pattern
- **1** = Severe anomaly (critical condition)

Higher scores indicate more abnormal patterns. The model will be evaluated on how well it identifies true anomalies in the test set.

---

## Project Overview

This solution implements a **multi-model ensemble approach** combining:
- Clinical domain knowledge (ICU medical thresholds)
- Machine learning anomaly detectors (Isolation Forest, OCSVM)
- Per-patient statistical analysis

### Input Data
| File | Purpose | Columns |
|------|---------|---------|
| **train_vitals.csv** | Training dataset | case_id, time_sec, HR, MBP, SpO2, Temp |
| **test_vitals.csv** | Test dataset | case_id, time_sec, HR, MBP, SpO2, Temp |

### Output
| File | Purpose | Format |
|------|---------|--------|
| **submissions.csv** | Anomaly predictions | case_id, time_sec, anomaly_score (0-1) |

---

##  Vital Signs Features

| Signal | Column | Unit | Normal Range | Critical Range |
|--------|--------|------|--------------|-----------------|
| **Heart Rate** | HR | bpm | 40–150 | <40 or >150 |
| **Mean Blood Pressure** | MBP | mmHg | 50–120 | <50 or >120 |
| **Oxygen Saturation** | SpO2 | % | 90–100 | <90 |
| **Temperature** | Temp | °C | 35.0–39.5 | <35.0 or >39.5 |

---

##  Getting Started: Download Dataset

### Option 1: Kaggle Competition (Recommended)
If this is a Kaggle competition dataset, download directly from the competition page:
1. Go to: **[Kaggle Competitions](https://kaggle.com/competitions)**
2. Search for the competition name
3. Download `train_vitals.csv` and `test_vitals.csv`
4. Place both files in the project directory

### Option 2: Local/Custom Dataset
If you have the dataset files:
1. Place `train_vitals.csv` in the project directory
2. Place `test_vitals.csv` in the project directory
3. Ensure column names match: `case_id, time_sec, HR, MBP, SpO2, Temp`

### Option 3: VitalDB Public Dataset
For reference data:
- **VitalDB**: [https://github.com/MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code)
- **MIMIC-III**: [https://mimic.physionet.org/](https://mimic.physionet.org/)

###  Project Directory Structure
```
anomaly detection/
├── Anomaly detection.ipynb  ← Main analysis notebook
├── train_vitals.csv         ← Download here
├── test_vitals.csv          ← Download here
├── submissions.csv          ← Generated output
└── README.md                ← This file
```

---

##  Quick Start

### 1. Prepare Data
```
Place train_vitals.csv and test_vitals.csv in the project folder
```

### 2. Run Notebook
```
Open Anomaly detection.ipynb in Jupyter and Run All Cells
```

### 3. Get Results
```
submissions.csv will be auto-generated with anomaly scores
```

### 4. Submit Results
```
Upload submissions.csv to the competition or evaluation platform
```

---

##  How Anomalies Are Detected

The model uses **5 independent detection methods** combined with weighted voting:

### 1️⃣ **Clinical Risk Scoring (Weight: 30%)**
- Applies hard ICU medical thresholds to vital signs
- Detects critical combinations (low SpO2 + high HR = respiratory distress)
- Most reliable signal (based on medical expertise)

### 2️⃣ **Per-Patient Deviation Analysis (Weight: 25%)**
- Compares readings against each patient's historical baseline
- Flags sudden changes in vital patterns for specific patients
- Key insight: Abnormal for *this patient* matters more than global norms

### 3️⃣ **Isolation Forest - Model 1 (Weight: 20%)**
- Unsupervised tree-based anomaly detection (500 trees)
- Learns non-linear patterns from training data
- Configuration: 5% contamination, max_samples=2048

### 4️⃣ **Isolation Forest - Model 2 (Weight: 15%)**
- Second independent tree model for robustness (300 trees)
- Configuration: 8% contamination, max_samples=4096
- Captures different aspects of data distribution

### 5️⃣ **One-Class SVM (Weight: 10%)**
- Support Vector Machine trained on 15K sample
- Learns boundary of normal data space
- RBF kernel captures local density patterns

**Final Prediction**:
```
anomaly_score = 0.30×Clinical + 0.25×Deviation + 0.20×IsoF1 + 0.15×IsoF2 + 0.10×SVM
                    ↓ All normalized to [0,1] ↓
                    Percentile-stretched output
```

---

## ⚙️ Feature Engineering

For machine learning models, creates **16+ derived features** per patient:

```
For each vital sign (HR, MBP, SpO2, Temp):
├── Raw value               (filled missing values)
├── Per-patient z-score     (deviation from patient mean)
├── 1-step change           (rate of change)
└── 5-step change           (trends over 5 time steps)
```

**Data Preprocessing**:
-  Forward-fill missing values within patient timelines
-  Fallback to patient median, then global median
-  Robust scaling (resistant to outliers)
-  Clip extreme values to [-5, +5] to prevent model corruption

**Why per-patient features matter**:
- Patient baseline variation is huge (young vs. elderly, healthy vs. critical)
- Relative change is better signal than absolute values
- Time-series context captures trending patterns

---

##  Calibration & Output Format

The model outputs **anomaly scores** normalized to [0, 1] range:

**submissions.csv**:
```csv
case_id,time_sec,anomaly_score
12345,0,0.125634
12345,1,0.089234
12346,0,0.823456
12346,1,0.156789
...
```

**Score Interpretation**:
| Range | Interpretation | Action |
|-------|-----------------|--------|
| 0.0 - 0.2 | Normal | Continue monitoring |
| 0.2 - 0.4 | Borderline | Review trend |
| 0.4 - 0.7 | Anomalous | Investigate cause |
| 0.7 - 1.0 | Critical Anomaly | Alert clinician immediately |

**Output Processing**:
1.  Min-Max Normalization → [0, 1] range
2.  Percentile Stretching → Better score separation
3.  Round to 6 decimals → Numerical precision

---

##  Pipeline Workflow

| Step | What Happens | Output |
|------|--------------|--------|
| 1. Load Data | Read train_vitals.csv & test_vitals.csv | DataFrames with vital sign columns |
| 2. Clinical Scoring | Apply ICU medical thresholds | Risk scores for each timestamp |
| 3. Feature Engineering | Create 16+ derived features per patient | Feature matrix |
| 4. Feature Scaling | Normalize with RobustScaler | Mean=0, robust to outliers |
| 5. Train IsoForest x2 | Learn normal data distribution (2 models) | 2 anomaly detectors |
| 6. Train OCSVM | Learn decision boundary on sample | SVM model |
| 7. Score All Methods | Get scores from 5 detection methods | 5 score vectors |
| 8. Ensemble Voting | Weighted combination of 5 scores | Single combined score |
| 9. Calibration | Percentile stretching to [0,1] | Final anomaly_score |
| 10. Output | Generate CSV submission | submissions.csv ready |

---

## Why This Approach Works

###  Strengths

| Feature | Benefit |
|---------|---------|
| **Clinical Domain Knowledge** | Captures medical expertise (30% weight → highest importance) |
| **Per-Patient Baseline** | Each patient has unique normal range — relative changes matter more |
| **Multi-Method Ensemble** | Different algorithms catch different anomaly types |
| **Automated Feature Engineering** | Captures temporal patterns (trends, changes, deviations) |
| **Robust Scaling** | Handles sensor noise and occasional spikes |

###  Challenges Handled

| Issue | Solution |
|-------|----------|
| Missing vital measurements | Forward-fill per patient, fallback to median |
| Patient baseline variation | Per-patient z-score normalization |
| Large dataset (500K+ rows) | Sample OCSVM (15K), limit tree samples |
| Score clustering | Percentile stretching for better separation |
| Extreme outliers | Clip features to [-5, +5] range |

---

##  Project Files

```
anomaly detection/
├── Anomaly detection.ipynb  ← Jupyter notebook with full solution
├── train_vitals.csv         ← Training data (download required)
├── test_vitals.csv          ← Test data (download required)
├── submissions.csv          ← Output: anomaly scores (auto-generated)
└── README.md                ← This guide
```

---

##  Technical Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Data Processing | pandas, numpy | Load, transform, aggregate vital signs |
| Feature Scaling | scikit-learn RobustScaler | Normalize features for ML models |
| Anomaly Detection | IsolationForest, OneClassSVM | Unsupervised learning models |
| Computation | Multi-threaded (n_jobs=-1) | Speed up training on large data |

**Performance**: Handles 500K+ rows efficiently

---

##  Expected Results

- **Score Range**: [0.0 (normal), 1.0 (critical anomaly)]
- **Typical Distribution**: Most readings 0.1-0.4 (normal), occasional spikes 0.7+
- **Evaluation**: Ranked by anomaly detection performance (accuracy, AUC, precision)

---

##  FAQ

**Q: What if data is missing?**
A: Forward-fill strategy fills gaps per patient, then uses median fallback.

**Q: How to improve results?**
A: Increase ensemble weight on clinical rules if you see false positives, or increase ML model weights for more sensitivity.

**Q: What's the computational time?**
A: Typically 5-15 minutes for 500K rows on standard machine with n_jobs=-1 (parallel).

---

##  Future Enhancements

- [ ] Add LSTM for temporal sequence modeling
- [ ] Incorporate patient metadata (age, diagnosis)
- [ ] Adaptive thresholds per patient demographics
- [ ] Real-time streaming predictions
- [ ] Interactive web dashboard for clinician review

---

**Last Updated**: April 2026  
**Problem Type**: Time-Series Anomaly Detection (ICU Medical)  
**Status**: Production-Ready ✅
