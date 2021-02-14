import argparse

from youtubesearchpython import VideosSearch
import os
from subprocess import check_call, CalledProcessError

if __name__ == "__main__":
    DESCRIPTION = 'Youtube Scrape'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    OPTIONAL_ARGUMENTS = PARSER.add_argument_group("optional arguments")
    REQUIRED_ARGUMENTS.add_argument('-search_word',
                                    action='store',
                                    help='Input search query for youtube ',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-path_to_save_audio_files',
                                    action='store',
                                    help='Input path to save audio files',
                                    required=True)

    OPTIONAL_ARGUMENTS.add_argument("-num_results", action="store",
                                    help="Number of audio clips to be downloaded",
                                    )
    RESULT = PARSER.parse_args()
    SEARCH_KEYWORD = RESULT.search_word
    PATH_TO_WRITE_AUDIO = RESULT.path_to_save_audio_files
    if RESULT.num_results:
        NUMBER_AUDIOCLIPS_LIMIT = (int)(RESULT.num_results)

    else:
        NUMBER_AUDIOCLIPS_LIMIT = 5
    ###########################################################################
    # create the directory path if not present
    ###########################################################################
    if not os.path.exists(PATH_TO_WRITE_AUDIO):
        os.mkdir(PATH_TO_WRITE_AUDIO)
    else:
        pass
    if not os.path.exists(PATH_TO_WRITE_AUDIO + "_".join(SEARCH_KEYWORD.split(" "))):
        os.mkdir(PATH_TO_WRITE_AUDIO + "_".join(SEARCH_KEYWORD.split(" ")))
    else:
        pass
    ###########################################################################
    # Settings for scraping the youtube webpage with selected query keyword
    ###########################################################################
    vid_search = VideosSearch(SEARCH_KEYWORD, limit=10)
    VIDEO_LIST = []
    for result_dict in vid_search.result()["result"]:
        VIDEO_LIST.append(result_dict["link"])

    ###########################################################################
    # Download audio file onto the target directory
    ###########################################################################
    COUNT = 0
    for item in VIDEO_LIST[:NUMBER_AUDIOCLIPS_LIMIT]:
        COUNT += 1
        path = PATH_TO_WRITE_AUDIO + "_".join(SEARCH_KEYWORD.split(" ")) + "/" + SEARCH_KEYWORD + "_" + item.split("=")[
            -1] + "_" + str(COUNT)
        if not os.path.exists(path):
            try:
                check_call(['youtube-dl', item,
                            '--audio-format', 'wav',
                            '-x', '-o', path + '.%(ext)s'])
            except CalledProcessError:

                # do nothing
                print("Exception CalledProcessError!")
        else:
            pass
