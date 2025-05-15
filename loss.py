#!/usr/bin/env python3

import pandas as pd
from sys import argv

def process(input):
    # Read the CSV data
    data = pd.read_csv(input)

    # Find the epoch with the lowest validation loss
    best_epoch = data.loc[data['val_loss'].idxmin()]

    # Write the best line (best epoch with lowest validation loss) to a new CSV file
    best_epoch.to_frame().T.to_csv('best_epoch.csv', index=False, mode='a', header=False)

for i in argv[1:]:
    process(i)
