# ============================================================
# ECG ARRHYTHMIA DETECTION
# FINAL EVALUATION
#
# Random Forest + ROC/AUC + Sensitivity + Specificity
# ============================================================

import wfdb
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# ============================================================
# 1. ECG RECORDS
# ============================================================

TRAIN_RECORDS = [
    "100",
    "101",
    "103",
    "105",
    "106",
    "107",
    "109",
    "111"
]

TEST_RECORDS = [
    "112",
    "113",
    "115",
    "116"
]


# ============================================================
# 2. PROCESS ECG RECORD
# ============================================================

def process_record(record_name):

    print("Processing record:", record_name)

    record = wfdb.rdrecord(
        record_name,
        pn_dir="mitdb"
    )

    ecg = record.p_signal[:, 0]

    fs = record.fs

    annotation = wfdb.rdann(
        record_name,
        "atr",
        pn_dir="mitdb"
    )


    # --------------------------------------------------------
    # Bandpass filter
    # --------------------------------------------------------

    low_cutoff = 0.5
    high_cutoff = 40

    b, a = butter(
        4,
        [
            low_cutoff / (fs / 2),
            high_cutoff / (fs / 2)
        ],
        btype="bandpass"
    )

    filtered_ecg = filtfilt(
        b,
        a,
        ecg
    )


    # --------------------------------------------------------
    # Heartbeat extraction
    # --------------------------------------------------------

    X = []
    y = []
    positions = []

    before = int(
        0.25 * fs
    )

    after = int(
        0.45 * fs
    )


    for sample, label in zip(
        annotation.sample,
        annotation.symbol
    ):

        # Only Normal and PVC
        if label not in ["N", "V"]:
            continue


        if sample - before < 0:
            continue


        if sample + after >= len(filtered_ecg):
            continue


        heartbeat = filtered_ecg[
            sample - before:
            sample + after
        ]


        if len(heartbeat) != before + after:
            continue


        X.append(
            heartbeat
        )

        positions.append(
            sample
        )


        if label == "N":
            y.append(0)

        elif label == "V":
            y.append(1)


    return (
        np.array(X),
        np.array(y),
        np.array(positions),
        filtered_ecg,
        fs
    )


# ============================================================
# 3. EXTRACT FEATURES
# ============================================================

def extract_features(
    X,
    positions,
    fs
):

    features = []


    for i, heartbeat in enumerate(X):

        # ----------------------------------------------------
        # Basic morphology
        # ----------------------------------------------------

        mean_amplitude = np.mean(
            heartbeat
        )

        max_amplitude = np.max(
            heartbeat
        )

        min_amplitude = np.min(
            heartbeat
        )

        std_amplitude = np.std(
            heartbeat
        )

        peak_to_peak = (
            max_amplitude
            - min_amplitude
        )

        energy = np.sum(
            heartbeat ** 2
        )

        rms = np.sqrt(
            np.mean(
                heartbeat ** 2
            )
        )


        # ----------------------------------------------------
        # Approximate peak width
        # ----------------------------------------------------

        threshold = (
            0.5 * max_amplitude
        )

        peak_width = np.sum(
            heartbeat > threshold
        )


        # ----------------------------------------------------
        # RR interval
        # ----------------------------------------------------

        if i == 0:

            rr_interval = 0

        else:

            rr_interval = (
                positions[i]
                - positions[i - 1]
            ) / fs


        # ----------------------------------------------------
        # Heart rate
        # ----------------------------------------------------

        if rr_interval > 0:

            heart_rate = (
                60 / rr_interval
            )

        else:

            heart_rate = 0


        features.append([

            mean_amplitude,

            max_amplitude,

            min_amplitude,

            std_amplitude,

            peak_to_peak,

            energy,

            rms,

            peak_width,

            rr_interval,

            heart_rate

        ])


    return np.array(
        features
    )


# ============================================================
# 4. BUILD TRAINING DATA
# ============================================================

print("\n")
print("=" * 60)
print("BUILDING TRAINING DATA")
print("=" * 60)


X_train_list = []
y_train_list = []
features_train_list = []


for record_name in TRAIN_RECORDS:

    (
        X_record,
        y_record,
        positions,
        filtered_ecg,
        fs
    ) = process_record(
        record_name
    )


    features_record = extract_features(
        X_record,
        positions,
        fs
    )


    X_train_list.append(
        X_record
    )

    y_train_list.append(
        y_record
    )

    features_train_list.append(
        features_record
    )


X_train = np.concatenate(
    X_train_list
)

y_train = np.concatenate(
    y_train_list
)

X_train_features = np.concatenate(
    features_train_list
)


# ============================================================
# 5. BUILD TESTING DATA
# ============================================================

print("\n")
print("=" * 60)
print("BUILDING TEST DATA")
print("=" * 60)


X_test_list = []
y_test_list = []
features_test_list = []


for record_name in TEST_RECORDS:

    (
        X_record,
        y_record,
        positions,
        filtered_ecg,
        fs
    ) = process_record(
        record_name
    )


    features_record = extract_features(
        X_record,
        positions,
        fs
    )


    X_test_list.append(
        X_record
    )

    y_test_list.append(
        y_record
    )

    features_test_list.append(
        features_record
    )


X_test = np.concatenate(
    X_test_list
)

y_test = np.concatenate(
    y_test_list
)

X_test_features = np.concatenate(
    features_test_list
)


# ============================================================
# 6. TRAIN ORIGINAL RANDOM FOREST
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)


rf_model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1
)


rf_model.fit(
    X_train_features,
    y_train
)


print(
    "Training complete!"
)


# ============================================================
# 7. PREDICTIONS
# ============================================================

predictions = rf_model.predict(
    X_test_features
)


# Probability of PVC

probabilities = rf_model.predict_proba(
    X_test_features
)[:, 1]


# ============================================================
# 8. BASIC PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)


accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print("\nClassification Report:")


print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "PVC"
        ],
        zero_division=0
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)


true_negative = cm[0, 0]

false_positive = cm[0, 1]

false_negative = cm[1, 0]

true_positive = cm[1, 1]


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


print(cm)


print("\nTrue Negatives:", true_negative)

print("False Positives:", false_positive)

print("False Negatives:", false_negative)

print("True Positives:", true_positive)


# ============================================================
# 10. SENSITIVITY
# ============================================================

# Sensitivity = TP / (TP + FN)

if (
    true_positive
    + false_negative
) > 0:

    sensitivity = (
        true_positive
        /
        (
            true_positive
            + false_negative
        )
    )

else:

    sensitivity = 0


# ============================================================
# 11. SPECIFICITY
# ============================================================

# Specificity = TN / (TN + FP)

if (
    true_negative
    + false_positive
) > 0:

    specificity = (
        true_negative
        /
        (
            true_negative
            + false_positive
        )
    )

else:

    specificity = 0


print("\n")
print("=" * 60)
print("SENSITIVITY AND SPECIFICITY")
print("=" * 60)


print(
    "Sensitivity:",
    round(
        sensitivity * 100,
        2
    ),
    "%"
)


print(
    "Specificity:",
    round(
        specificity * 100,
        2
    ),
    "%"
)


# ============================================================
# 12. ROC CURVE
# ============================================================

false_positive_rate, true_positive_rate, thresholds = (
    roc_curve(
        y_test,
        probabilities
    )
)


# ============================================================
# 13. AUC
# ============================================================

auc_score = roc_auc_score(
    y_test,
    probabilities
)


print("\n")
print("=" * 60)
print("ROC / AUC")
print("=" * 60)


print(
    "AUC:",
    round(
        auc_score,
        4
    )
)


# ============================================================
# 14. PLOT ROC CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.plot(
    false_positive_rate,
    true_positive_rate,
    color="blue",
    linewidth=2,

    label=(
        "Random Forest "
        "AUC = "
        + str(round(auc_score, 3))
    )
)


# Random guessing line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random classifier"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve - PVC Detection"
)


plt.legend()

plt.grid(
    alpha=0.3
)


plt.tight_layout()

plt.show()


# ============================================================
# 15. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm,
    cmap="Blues"
)


plt.title(
    "Random Forest Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.xticks(
    [0, 1],
    ["Normal", "PVC"]
)


plt.yticks(
    [0, 1],
    ["Normal", "PVC"]
)


plt.colorbar()


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )


plt.tight_layout()

plt.show()


# ============================================================
# 16. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL ECG ARRHYTHMIA DETECTION RESULTS")
print("=" * 60)


print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print(
    "Sensitivity:",
    round(
        sensitivity * 100,
        2
    ),
    "%"
)


print(
    "Specificity:",
    round(
        specificity * 100,
        2
    ),
    "%"
)


print(
    "AUC:",
    round(
        auc_score,
        4
    )
)


print(
    "PVCs detected:",
    true_positive
)


print(
    "PVCs missed:",
    false_negative
)


print(
    "False PVC predictions:",
    false_positive
)


print("\n")
print("=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)