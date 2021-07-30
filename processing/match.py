import argparse
import datetime
import os
from tqdm import tqdm
import pandas as pd
import csv

#
# Constants
#

LONG_BED_TO_SHORT_BED = {
    "ALPHA": "A",
    "BRAVO": "B",
    "CARD": "C",
    "DELTA": "D",
    "FAST": "FT"
}

#
# Main Logic
#
def run(cohort_file, experiment_folders, cohort_output_file, export_output_file):
    df = pd.read_csv(cohort_file)
    print(f"Read cohort file: {df.shape}")

    # Removes patients who moved to a different bed
    rows_to_keep = []
    for i, row in df.iterrows():
        if row["First_bed"] == row["Last_bed"]:
            rows_to_keep.append(True)
        elif row["Last_bed"] == "ADULT ED OVRFLW":
            rows_to_keep.append(True)
        else:
            rows_to_keep.append(False)
    df = df[rows_to_keep]
    print(f"After eliminating patients who moved beds: {df.shape}")

    # Remove rows where important times are Nan
    df = df[df['Dispo_time'].notna()]
    df = df[df['Arrival_time'].notna()]
    print(f"After eliminating patients who had invalid times: {df.shape}")

    # Read bed files
    export_files = []
    for experiment_folder in experiment_folders:
        for filename in os.listdir(experiment_folder):
            if filename.endswith(".csv"):
                export_files.append(os.path.join(experiment_folder, filename))
    print(f"Reading the following experiment summary files: {export_files}")

    # Transform the bed labels
    export_dfs = []
    for export_file in export_files:
        df_export = pd.read_csv(export_file)
        bed_transformed = []
        for i, row in df_export.iterrows():
            orig_bed_label = row["BedLabel"]
            bed_number = orig_bed_label[0:2]
            bed_room = orig_bed_label[2:]
            if bed_room in LONG_BED_TO_SHORT_BED:
                short_bed_room = LONG_BED_TO_SHORT_BED[bed_room]
                bed_transformed.append(f"{short_bed_room}{bed_number}")
            else:
                bed_transformed.append(f"UNKNOWN")

        df_export["BedLabel_Transformed"] = bed_transformed
        export_dfs.append(df_export)

    # Remove unknown bed labels
    for i in range(export_dfs):
        print(f"Removing unknown bed labels of shape: {export_dfs[i][export_dfs[i]['BedLabel_Transformed'] == 'UNKNOWN'].shape}")
        export_dfs[i] = export_dfs[i][export_dfs[i]["BedLabel_Transformed"] != "UNKNOWN"]

    # Add directory label to each export summary file
    for i, export_df in enumerate(export_dfs):
        export_path = "/".join(export_files[i].split("/")[:-1])
        export_df["path"] = export_path

    # Merge dataframes
    final_export_df = pd.concat(export_dfs, ignore_index=True)
    print(f"Total export file combined: {final_export_df.shape}")

    # Remove beds with bad start/end times
    final_export_df = final_export_df[final_export_df['StartTime'].notna()]
    final_export_df = final_export_df[final_export_df['EndTime'].notna()]
    print(f"After removing beds with bad times: {final_export_df.shape}")
    final_export_df["final_export_df_id"] = range(len(final_export_df))

    # Drop duplicate values
    final_export_df = final_export_df.drop_duplicates(subset='StudyId')
    print(f"After removing duplicate studies: {final_export_df.shape}")

    # Get a map of the bed start times
    #
    print(f"Starting matching process...")
    start_time_to_export_row = {}
    start_end_times = []

    # Find the corresponding study folder(s)
    for j, export_row in final_export_df.iterrows():
        bed_start_time = export_row["StartTime"]
        bed_end_time = export_row["EndTime"]
        bed_label = export_row["BedLabel_Transformed"]
        bed_start_time = datetime.datetime.strptime(bed_start_time, "%m/%d/%y %H:%M:%S").replace(tzinfo=None)
        bed_end_time = datetime.datetime.strptime(bed_end_time, "%m/%d/%y %H:%M:%S").replace(tzinfo=None)

        if bed_start_time not in start_time_to_export_row:
            start_time_to_export_row[bed_start_time] = {}
        start_time_to_export_row[bed_start_time][bed_label] = export_row

        start_end_times.append((bed_start_time, bed_end_time, bed_label))

    # For each patient visit in the cohort file...
    matched = []
    final_export_df_ids = []
    final_studies = []

    final_export_df_id_to_case_id = {}
    case_id_to_export_rows = {}

    for i, row in tqdm(df.iterrows()):
        roomed_time = row["Roomed_time"]
        roomed_time = datetime.datetime.strptime(roomed_time, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = row["Dispo_time"]
        dispo_time = datetime.datetime.strptime(dispo_time, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        bed = row["First_bed"]
        case_id = row["CSN"]
        studies = []
        export_df_ids = []

        # A patient can span more than one study. The following code ensures that we can properly
        # merge the studies together.
        #
        # PATIENT:         ******
        # STUDY1 :     ------
        # STUDY2 :           --
        # STUDY3 :             ----
        # STUDY4 :        -----------
        found = False
        rows_found = []
        for start_end_time in start_end_times:
            if roomed_time <= start_end_time[1] and dispo_time >= start_end_time[0] and bed == start_end_time[2]:
                export_row = start_time_to_export_row[start_end_time[0]][start_end_time[2]]
                export_df_ids.append(str(export_row["final_export_df_id"]))
                studies.append(export_row["StudyId"])
                found = True
                rows_found.append(export_row)
                if case_id not in case_id_to_export_rows:
                    case_id_to_export_rows[case_id] = []
                case_id_to_export_rows[case_id].append(export_row.tolist())

        if not found:
            matched.append(False)
            final_export_df_ids.append("UNKNOWN")
        else:
            matched.append(True)
            final_export_df_ids.append(",".join(export_df_ids))
            for export_df_id in export_df_ids:
                final_export_df_id_to_case_id[export_df_id] = case_id

        final_studies.append(",".join(studies))
    print(f"A total of {len(case_id_to_export_rows)} cases have matching beds")

    # Create the matched export file which contains matched beds
    with open(export_output_file, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')
        headers = final_export_df.columns.tolist()
        headers.append("CSN")
        writer.writerow(headers)
        for case_id, rows in case_id_to_export_rows.items():
            for row in rows:
                output_row = row
                output_row.append(case_id)
                writer.writerow(output_row)

    # Create the matched cohort file which contains matched patient visits
    df_cohort_matched = df.copy()
    df_cohort_matched["study_matched"] = matched
    df_cohort_matched["final_export_df_ids"] = final_export_df_ids
    df_cohort_matched["final_studies"] = final_studies
    df_cohort_matched = df_cohort_matched[df_cohort_matched["study_matched"] == True]
    df_cohort_matched.to_csv(cohort_output_file)


#
# Main
#
parser = argparse.ArgumentParser(description='Matches a cohort file with a Philips bed summary file')
parser.add_argument('-ci', '--cohort-input-file',
                    required=True,
                    help='The path to a cohort file. e.g. "/deep/group/ed-monitor/cp_cohort_v3.csv"')
parser.add_argument('-ef', '--experiment-folders',
                    required=True,
                    help='Comma separated paths containing the experiment directories. e.g. "/deep/group/ed-monitor/2020_08_23_2020_09_23,/deep/group/ed-monitor/2020_09_23_2020_11_30"')
parser.add_argument('-co', '--cohort-output-file',
                    required=True,
                    help='The path to the cohort output file. e.g. "/deep/group/ed-monitor/cohort.output.csv"')
parser.add_argument('-eo', '--export-output-file',
                    required=True,
                    help='The path to the export output file. e.g. "/deep/group/ed-monitor/export.output.csv"')

args = parser.parse_args()

run(args.cohort_input_file, args.experiment_folders, args.cohort_output_file, args.export_output_file)
