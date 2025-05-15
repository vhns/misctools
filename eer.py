#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

def gen_y_true(csv_file):
    out = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file, delimiter=' ')
        for img, label in reader:
            out.append(int(label))
    return out

def gen_y_score(npfile):
    out = np.load(npfile)
    return out[:,1]

def calculate_eer(y_true, y_scores):
    """
    Calculate the Equal Error Rate (EER) for binary classification.
    
    Args:
        y_true: True labels (binary). [0]
        y_scores: Probabilities for the positive class (class B).
        
    Returns:
        EER threshold and the corresponding EER value.
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_index]
    eer_value = fpr[eer_index]  # fpr == fnr at EER
    return eer_threshold, eer_value

def generate_eer_report(y_true, y_scores, output_csv='eer_report.csv', output_plot='eer_plot.png'):
    """
    Generate the EER report, both as a CSV and a matplotlib plot.
    
    Args:
        y_true: True binary labels.
        y_scores: Probabilities of class B.
        output_csv: File name for the CSV report.
        output_plot: File name for the matplotlib plot.
    """
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # Calculate the EER
    eer_threshold, eer_value = calculate_eer(y_true, y_scores)
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    
    # Save CSV report
    eer_data = {
        'Threshold': thresholds,
        'FPR': fpr,
        'TPR': tpr,
        'FNR': fnr
    }
    df = pd.DataFrame(eer_data)
    df.to_csv(output_csv, index=False)
    print(f"EER report saved as {output_csv}")
    
    # Plot ROC curve with EER point
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot(fpr, fnr, label='FNR Curve', linestyle='--')
    plt.plot(fpr[eer_index], tpr[eer_index], 'ro', label=f'EER Point (Threshold={eer_threshold:.2f}, EER={eer_value*100:.2f}%)')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_plot, format='svg')
    print(f"EER plot saved as {output_plot}")

# Example usage:
# Assuming y_true (true labels) and y_scores (classifier probabilities) are given:
# y_true = np.array([...])  # True binary labels (0 or 1)
# y_scores = np.array([...])  # Probabilities for the positive class (class B)

# generate_eer_report(y_true, y_scores)

if __name__ == '__main__':
    args = argparse.ArgumentParser(
        prog='Foo',
        description='Bar')

    args.add_argument('--csv', type=str, required=True)
    args.add_argument('--npy', type=str, required=True)
    args.add_argument('--output-csv', type=str, required=True)
    args.add_argument('--output', type=str, required=True)

    args = args.parse_args()

    y_true = gen_y_true(args.csv)
    y_scores = gen_y_score(args.npy)

    generate_eer_report(y_true,y_scores, args.output_csv, args.output)
