"""
Downloads Sounds of Interest audio files amoung the list
"""
import argparse
import sys
from youtube_audioset import download_data
from youtube_audioset import EXPLOSION_SOUNDS, WOOD_SOUNDS, NATURE_SOUNDS
from youtube_audioset import MOTOR_SOUNDS, HUMAN_SOUNDS, TOOLS_SOUNDS
from youtube_audioset import DOMESTIC_SOUNDS, WILD_ANIMALS

###############################################################################
# create a dictionary of sounds
###############################################################################

SOUNDS_DICT = {'explosion_sounds': EXPLOSION_SOUNDS, 'wood_sounds': WOOD_SOUNDS,
               'nature_sounds': NATURE_SOUNDS, 'motor_sounds': MOTOR_SOUNDS,
               'human_sounds': HUMAN_SOUNDS, 'tools': TOOLS_SOUNDS,
               'domestic_sounds': DOMESTIC_SOUNDS, 'Wild_animals': WILD_ANIMALS}

if __name__ == '__main__':

    ###########################################################################
    # Description and Help
    ###########################################################################

    DESCRIPTION = 'Input one of these sounds : explosion_sounds , wood_sounds , motor_sounds,\
                   human_sounds, tools ,domestic_sounds, Wild_animals, nature_sounds'
    HELP = 'Input the target sounds. It should be one of the listed sounds'

    ###########################################################################
    # parse the input arguments given from command line
    ###########################################################################
    ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_NAMED = ARGUMENT_PARSER._action_groups.pop()

    REQUIRED_NAMED = ARGUMENT_PARSER.add_argument_group('required arguments')
    REQUIRED_NAMED.add_argument('-target_sounds', '--target_sounds', action='store',
                        help=HELP, required=True)
    REQUIRED_NAMED.add_argument('-target_path', '--target_path', action='store',
                        help='Input the path',
                        required=True)
    OPTIONAL_NAMED.add_argument('-file_limit', '--file_limit', type=int,
                                help='limit of files to download',
                                default=sys.maxsize)
    ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
    RESULT = ARGUMENT_PARSER.parse_args()

    download_data(aggregate_name=RESULT.target_sounds,
                  target_sounds_list=SOUNDS_DICT[RESULT.target_sounds],
                  target_path=RESULT.target_path,
                  file_limit=RESULT.file_limit)
