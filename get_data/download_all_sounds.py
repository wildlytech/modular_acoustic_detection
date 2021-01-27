"""
Downloads all the Impact and Ambient sounds
"""
import argparse
import sys
from youtube_audioset import download_data, AMBIENT_SOUNDS, IMPACT_SOUNDS

###############################################################################
# Main Function
###############################################################################
if __name__ == "__main__":

    ###########################################################################
    # Description and Help
    ###########################################################################

    DESCRIPTION = "Downloads all sounds"
    HELP = "Data can be huge. Use file_limit arg to limit number of files" \
           " downloaded."

    ###########################################################################
    # parse the input arguments given from command line
    ###########################################################################
    ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_NAMED = ARGUMENT_PARSER._action_groups.pop()
    REQUIRED_NAMED = ARGUMENT_PARSER.add_argument_group('required arguments')
    OPTIONAL_NAMED.add_argument('-file_limit', '--file_limit', type=int,
                                help='limit of files to download',
                                default=sys.maxsize)
    ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
    RESULT = ARGUMENT_PARSER.parse_args()

    TARGET_SOUNDS = AMBIENT_SOUNDS + IMPACT_SOUNDS
    TARGET_PATH = 'sounds/'
    download_data(aggregate_name="all",
                  target_sounds_list=TARGET_SOUNDS,
                  target_path=TARGET_PATH,
                  file_limit=RESULT.file_limit)
