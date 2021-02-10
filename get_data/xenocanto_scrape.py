import requests
import youtube_dl
import argparse

api_link = "https://www.xeno-canto.org/api/2/recordings?query="


###############################################################################


def query_api(bird_species):
    res = requests.get(api_link + bird_species)
    recordings = res.json()["recordings"]
    links = [recordings[i]["file"] for i in range(len(recordings))]
    ids = ["XC" + recordings[i]["id"] for i in range(len(recordings))]
    return links, ids


def download_audio(audio_links, audio_files_path, bird_ids):
    if not audio_files_path.endswith("/"):
        audio_files_path += '/'

    for link, ids in zip(audio_links, bird_ids):
        print("L: ", link)
        ydl_opts = {'outtmpl': audio_files_path + ids + '.%(ext)s'}

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])


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

    links, ids = query_api(BIRD_SPECIES_KEYWORD)

    download_audio(links, AUDIO_FILES_PATH, ids)
