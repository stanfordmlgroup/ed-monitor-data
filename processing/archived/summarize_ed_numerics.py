#!/usr/bin/env python

"""
Run after the prepare_ed_numerics_from_matched_cohort.py script to create a file that summarizes which CSNs were 
processed into a pickle file. 

Files are written out to:
- /deep/group/physiologic-states/v1/processed/<hash>/<CSN>.pkl
where <hash> is the last two characters of the CSN

Example: python summarize_ed_numerics.py -i /deep/group/physiologic-states/v1/processed
"""    

import os
import csv
import argparse
import datetime
import pickle
from tqdm import tqdm

def run(args):
    print(f"START TIME: {datetime.datetime.now()}")
    input_folder = args.input_folder

    cols = ["HR", "RR", "SpO2", "NBPs", "NBPd", "btbRRInt_ms", "Perf"]
    rows = []
    for item in tqdm(os.listdir(input_folder)):
        if not os.path.isfile(os.path.join(input_folder, item)):
            for csn in os.listdir(f"{input_folder}/{item}"):
                row = [csn]

                for fn_fn in cols:
                    num_rows = 0
                    try:
                        with open(f"{input_folder}/{item}/{csn}/{fn_fn}.csv", 'r') as handle:
                            for line in handle:
                                num_rows += 1
                        num_rows = num_rows - 1 # minus one to account for header
                        row.append(str(num_rows))
                    except:
                        row.append("0")
                rows.append(row)

    with open(f"{input_folder}/summary.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        header = ["csn"]
        for col in cols:
            header.append(f"{col}_len")
        writer.writerow(header)
        for line in rows:
            writer.writerow(line)

    print(f"END TIME: {datetime.datetime.now()}")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarizes what ED numerics files are available')
    parser.add_argument('-i', '--input-folder',
                        required=True,
                        help='Folder containing the individual hash folders')

    args = parser.parse_args()

    run(args)

    print("DONE")
