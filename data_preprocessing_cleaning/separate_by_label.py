'''
copy the desired label row of given csv file to a new csv file
'''
import argparse
import pandas as pd
import csv


def separate_by_label(path, label_to_separate):
    '''Copy the desired label row from given csv file to a new csv file'''
    # read the csv file
    csv_df = pd.read_csv(path, error_bad_lines=False)
    # get the required columns from the dataframe and create new dataframe
    new_csv_df = csv_df[['wav_file', 'Label_1', 'Label_2', 'Label_3', 'Label_4']]
    # column names for csv file
    column_tags = ['wav_file', 'Label_1', 'Label_2', 'Label_3', 'Label_4']
    old_csv_file = path.split("/")[-1]
    new_csv_file = ".".join(old_csv_file.split(".")[:-1])+"_"+label_to_separate.replace(' ', '_')+'.csv'
    print("\nNew CSV file:", new_csv_file)

    # create a csv new file to export data
    with open(new_csv_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_tags)

        for row in range(0, new_csv_df.shape[0]):
            for column in range(1, len(column_tags)):
                if new_csv_df.iloc[row][column] == label_to_separate:
                    csvwriter.writerow(new_csv_df.iloc[row])
                    break

###############################################################################

if __name__ == "__main__":

    DESCRIPTION = 'Copy the desired label row from given csv file and export to a new CSV file'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    RequiredArguments = PARSER.add_argument_group('required arguments')
    RequiredArguments.add_argument('-csv', '--csv_file_path', action='store', \
        help='Input csv file path', required=True)
    RequiredArguments.add_argument('-label', '--label_to_separate', action='store', \
        help='Input label to separate from csv file, for instance "bird"', required=True)
    RESULT = PARSER.parse_args()

    print("\nGiven CSV file path:", RESULT.csv_file_path)
    print("\nGiven label to separate from the csv file:", RESULT.label_to_separate)

    separate_by_label(RESULT.csv_file_path, RESULT.label_to_separate)
