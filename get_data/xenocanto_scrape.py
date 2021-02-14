import argparse
import os
import pandas as pd
import requests
import youtube_dl

api_link = "https://www.xeno-canto.org/api/2/recordings?query="


###############################################################################


def query_api(bird_species):
    """
    Query xeno-canto api for links to audio for bird species
    """

    res = requests.get(api_link + bird_species)
    recordings = res.json()["recordings"]

    df = pd.DataFrame(recordings)

    # The following columns are maintained for legacy purposes when the html
    # was being scraped
    df["XenoCanto_ID"] = "XC" + df['id']

    scientific_name = df["gen"] + " " + df["sp"] + " " + df["ssp"]
    # Strip trailing whitespaces in case there is no subspecies identifier
    scientific_name = scientific_name.str.strip()
    df["Common name/Scientific"] = df["en"] + " (" + scientific_name + ")"

    df["Length"] = df["length"]
    df["Recordist"] = df["rec"]
    df["Date"] = df["date"]
    df["Time"] = df["time"]
    df["Country"] = df["cnt"]
    df["Location"] = df["loc"]
    df["Elev(m)"] = df["alt"]
    df["Type"] = df["type"]
    df["Remarks"] = df["rmk"]

    return df


def download_xc_audio(audio_files_path, xc_audio_ID):
    """
    Download audio pertaining to particular xenocanto ID
    """

    # Append slash if directory doesn't have it already
    if not audio_files_path.endswith("/"):
        audio_files_path += '/'

    # download xc audio file in the given path
    # Link to audio does not contain first two letters of ID (Typically 'XC')
    audio_link = 'https://www.xeno-canto.org/' + xc_audio_ID[2:] + '/download'
    print(audio_link)
    ydl_opts = {'outtmpl': audio_files_path + xc_audio_ID + '.%(ext)s'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([audio_link])


def scrape(audio_files_path, bird_species):
    """
    Scrape all audio files for a particular bird species
    """

    # Append slash if directory doesn't have it already
    if not audio_files_path.endswith("/"):
        audio_files_path += '/'

    print("\nBird species keyword:", bird_species)
    print("Audio files path:", audio_files_path, '\n')

    bird_species = bird_species.lower()
    # replace whitespace with underscore for bird_species name
    bird_species_name_ws = bird_species.replace(' ', '_')

    dir_path = audio_files_path + bird_species_name_ws + '/'
    # if not exists create directory with bird species name to save audio files
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    query_results_df = query_api(bird_species)

    # Download all files specified in query results
    for row in query_results_df.iterrows():
        if not os.path.isfile(dir_path + row[1]["XenoCanto_ID"] + ".mp3"):
            # download the audio file
            download_xc_audio(dir_path, row[1]["XenoCanto_ID"])

    # csv file name appended with bird species
    csv_filename = dir_path + "xenocanto_bird_" + bird_species_name_ws + ".csv"
    print("csv file path:", csv_filename)

    # For existing entries, if they haven't already been covered by the
    # results of the query, add it and download it
    if os.path.exists(csv_filename):
        xc_csv = pd.read_csv(csv_filename, error_bad_lines=False)

        for row in xc_csv.iterrows():
            existing_ids = query_results_df["XenoCanto_ID"].values
            if row[1]["XenoCanto_ID"] not in existing_ids:
                query_results_df = query_results_df.append(row[1])

            if not os.path.isfile(dir_path + row[1]["XenoCanto_ID"] + ".mp3"):
                # download the audio file
                download_xc_audio(dir_path, row[1]["XenoCanto_ID"])

    query_results_df.to_csv(csv_filename, index=False)

    return dir_path, csv_filename


########################################################################
# Main Function
########################################################################
if __name__ == "__main__":
    DESCRIPTION = 'Scrape XenoCanto'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-bird_species',
                                    action='store',
                                    help='Input bird species by separating name with \
                                    space and enclosed within quotes, \
                                    for instance "ashy prinia" ',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-path_to_save_audio_files',
                                    action='store',
                                    help='Input path to save audio files',
                                    required=True)
    RESULT = PARSER.parse_args()

    BIRD_SPECIES_KEYWORD = RESULT.bird_species
    AUDIO_FILES_PATH = RESULT.path_to_save_audio_files

    scrape(audio_files_path=AUDIO_FILES_PATH,
           bird_species=BIRD_SPECIES_KEYWORD)
