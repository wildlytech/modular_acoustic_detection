from youtubesearchpython import VideosSearch
import os
from subprocess import check_call, CalledProcessError

###########################################################################
# Change as per the need
###########################################################################
SEARCH_KEYWORD = "dog barking"
PATH_TO_WRITE_AUDIO = "youtube_scraped_audio/"
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

print("Links: ", VIDEO_LIST)
###########################################################################
# Download audio file onto the target directory
###########################################################################
COUNT = 0
for item in VIDEO_LIST[:NUMBER_AUDIOCLIPS_LIMIT]:
    COUNT += 1
    path = PATH_TO_WRITE_AUDIO + "_".join(SEARCH_KEYWORD.split(" ")) + "/" + SEARCH_KEYWORD + "_" + item.split("=")[-1] + "_" + str(COUNT)
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
