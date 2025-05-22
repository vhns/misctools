#!/usr/bin/env python3

import numpy as np
import sys

if len(sys.argv) != 4:
    print("Usage: python accuracy.py true.npy prediction.npy output.txt")
    sys.exit(1)

true_path = sys.argv[1]
pred_path = sys.argv[2]
output_path = sys.argv[3]

# Load the arrays
true = np.load(true_path)
prediction = np.load(pred_path)

# Calculate accuracy
accuracy = np.mean(true == prediction)

# Save accuracy to the output file
with open(output_path, "w") as f:
    f.write(f"{accuracy:.2f}")
