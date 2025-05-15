#!/usr/bin/env python3

import csv
import argparse
from random import shuffle

if __name__ == "__main__":
    args = argparse.ArgumentParser(
        prog='splits input csv in equal amounts of classes',
        description='Does what it says on the tin according to given parameters')

    args.add_argument("--csv", type=str, required=True)
    args.add_argument("--class-pos", type=int, required=True)
    args.add_argument("--output", type=str, required=True)
    args.add_argument("--amount", type=int, required=False, default=None)

    args = args.parse_args()

    pos = args.class_pos
    classes = {}

    with open(args.csv, mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for words in reader:
            current_class = words[pos]
            if current_class in classes:
                classes[current_class] += 1
            else:
                classes[current_class] = 1

    if not args.amount:
        equal = min([i for i in classes.values()])
    else:
        equal = args.amount

    for i in classes:
        classes[i] = equal

    lines = [line.rstrip() for line in open(args.csv, mode='r')]
    out = []
    shuffle(lines)

    reader = csv.reader(lines, delimiter =' ')
    for line in reader:
        current_class = line[pos]
        if classes[current_class] >= 1:
            classes[current_class] -= 1
            out.append(" ".join(line))

    shuffle(out)

    with open(args.output, mode='x') as out_file:
        writer = csv.writer(out_file, delimiter=' ')
        reader = csv.reader(out, delimiter =' ')
        for line in reader:
            writer.writerow(line)
