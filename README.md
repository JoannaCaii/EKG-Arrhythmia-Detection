# EKG-Arrhythmia-Detection
EKG PVC detection using Random Forest and MIT-BIH Arrhythmia Database

EKG Arrhythmia Detection Using Random Forest

A machine learning project for detecting Premature Ventricular Contractions (PVCs) from EKG signals using the MIT-BIH Arrhythmia Database.

Overview

This project processes EkG recordings, extracts heartbeat-level features, and uses a Random Forest classifier to distinguish between:

Normal heartbeats

PVC heartbeats

The model is evaluated using:

Accuracy

Sensitivity

Specificity

Confusion matrix

ROC curve

AUC

Dataset

The project uses the MIT-BIH Arrhythmia Database through the WFDB Python package.

The EkG records are divided into training and testing sets.

Training records
100, 101, 103, 105, 106, 107, 109, 111

Testing records
112, 113, 115, 116

Signal Processing

Each EKG record is processed using:

EkG signal loading with WFDB

A 4th-order Butterworth bandpass filter

A frequency range of 0.5–40 Hz

Heartbeat extraction using annotation locations

Selection of Normal (N) and PVC (V) beats

Features

The following features are extracted from each heartbeat:

Mean amplitude

Maximum amplitude

Minimum amplitude

Standard deviation

Peak-to-peak amplitude

Signal energy

RMS

Approximate peak width

RR interval

Heart rate

Machine Learning

A Random Forest classifier is trained with:

100 trees

Balanced class weights

Random state of 42

Evaluation

The model is evaluated on EKG records that are separate from the training records.

The following metrics are reported:

Accuracy
Sensitivity
Specificity
ROC-AUC


A confusion matrix and ROC curve are also generated.

How to Run

Clone the repository:

git clone https://github.com/YOUR-USERNAME/ekg-arrhythmia-detection.git
cd ekg-arrhythmia-detection


Install the dependencies:

pip install -r requirements.txt


Run the program:

python ekg_arrhythmia_detection.py


The program downloads the required MIT-BIH records through WFDB and produces the evaluation results and plots.

Project Structure
ekg-arrhythmia-detection/
│
├── README.md
├── requirements.txt
├── ekg_arrhythmia_detection.py
├── results/
│   ├── roc_curve.png
│   └── confusion_matrix.png
└── .gitignore

Disclaimer

This project is intended for educational and research purposes. It is not a medical diagnostic system and should not be used for clinical decision-making.
