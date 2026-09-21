# EKG Arrhythmia Detection

EKG PVC detection using Random Forest and the MIT-BIH Arrhythmia Database.

## Overview

This project uses machine learning to detect **Premature Ventricular Contractions (PVCs)** from EKG signals using the MIT-BIH Arrhythmia Database.

The project processes EKG recordings, extracts heartbeat-level features, and uses a Random Forest classifier to distinguish between:

- Normal heartbeats
- PVC heartbeats

The model is evaluated using:

- Accuracy
- Sensitivity
- Specificity
- Confusion matrix
- ROC curve
- ROC-AUC

## Dataset

The project uses the **MIT-BIH Arrhythmia Database** through the WFDB Python package.

The EKG records are divided into separate training and testing sets.

### Training Records

100, 101, 103, 105, 106, 107, 109, 111

### Testing Records 
112, 113, 115, 116

## Signal Processing
Each EKG recording is processed using the following steps:

1. Load the EKG signal using WFDB.

2. Apply a 4th-order Butterworth bandpass filter.

3. Use a frequency range of 0.5–40 Hz.

4. Use the provided EKG annotations to identify individual heartbeats.

5. Select heartbeats labeled:

     N — Normal

     V — Premature Ventricular Contraction (PVC)

6. Extract a fixed window around each annotated heartbeat.

## Feature Extraction
The following features are extracted from each heartbeat:

## Feature Extraction

The following features are extracted from each heartbeat:

| Feature | Description |
|---|---|
| Mean Amplitude | Average signal amplitude |
| Maximum Amplitude | Maximum value in the heartbeat |
| Minimum Amplitude | Minimum value in the heartbeat |
| Standard Deviation | Variation in signal amplitude |
| Peak-to-Peak | Difference between maximum and minimum amplitude |
| Energy | Sum of squared signal values |
| RMS | Root mean square of the signal |
| Peak Width | Approximate width based on an amplitude threshold |
| RR Interval | Time between consecutive detected beats |
| Heart Rate | Estimated heart rate from the RR interval |


## Machine Learning
A Random Forest Classifier is used to classify the extracted heartbeat features.

## Model Parameters
Number of Trees: 100
Class Weight: Balanced
Random State: 42

The class_weight="balanced" setting is used to account for differences in the number of Normal and PVC heartbeats.

## Model Evaluation
The model is evaluated using testing records that were not used during training.

## Accuracy
Accuracy measures the overall proportion of correctly classified heartbeats.

## Sensitivity
Sensitivity measures the proportion of actual PVC heartbeats that are correctly detected.

Sensitivity = TP / (TP + FN)

## Specificity
Specificity measures the proportion of Normal heartbeats that are correctly identified.

Specificity = TN / (TN + FP)

## ROC-AUC
A Receiver Operating Characteristic (ROC) curve is generated using the model's predicted PVC probabilities.

The Area Under the Curve (AUC) is calculated to measure how well the model distinguishes between Normal and PVC heartbeats across different classification thresholds.

## Confusion Matrix
The confusion matrix shows:

  True Negatives (TN)
  False Positives (FP)
  False Negatives (FN)
  True Positives (TP)

## Results
The program calculates and displays the following performance metrics:

Accuracy
Sensitivity
Specificity
ROC-AUC
True Negatives
False Positives
False Negatives
True Positives

The program also generates:

  ROC Curve
  Confusion Matrix

## ROC Curve
The ROC curve shows the relationship between the true positive rate and false positive rate at different classification thresholds.

## Confusion Matrix
The confusion matrix provides a breakdown of the model's predictions for Normal and PVC heartbeats.

## How to Run
1. Clone the Repository
  git clone https://github.com/JoannaCaii/EKG-Arrhythmia-Detection.git
  cd EKG-Arrhythmia-Detection

2. Install the Dependencies
Install the required Python packages using:

pip install -r requirements.txt

The project uses the following Python libraries:

wfdb
numpy
matplotlib
scipy
scikit-learn

3. Run the Program
Run the Python script:

python EKG_Arrhythmia_Detection.py

You can also open the project in PyCharm and run the Python file using the Run button.

4. View the Results
After the program finishes, it will display:

Accuracy
Sensitivity
Specificity
ROC-AUC
Confusion Matrix
ROC Curve

The MIT-BIH EKG records are accessed through the WFDB Python package.

## Project Structure
EKG-Arrhythmia-Detection/
│
├── README.md
├── requirements.txt
├── EKG_Arrhythmia_Detection.py
├── results/
│   ├── roc_curve.png
│   └── confusion_matrix.png
└── .gitignore

## Technologies Used
Python
WFDB
NumPy
SciPy
Matplotlib
Scikit-learn
Random Forest
Digital Signal Processing

## Future Improvements
Potential improvements to this project include:

Adding additional EKG features

Improving heartbeat segmentation

Testing additional machine learning algorithms

Comparing Random Forest with other classifiers

Adding additional heartbeat classes

Improving RR interval feature calculation

Adding automated result visualization

Testing the model on additional EKG records

## Disclaimer
This project is intended for educational and research purposes only.

It is not a medical diagnostic system and should not be used for clinical decision-making.

## Author
Joanna Cai
GitHub: https://github.com/JoannaCaii

