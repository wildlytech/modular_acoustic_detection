from bs4 import BeautifulSoup as bs
import requests
import os
from subprocess import check_call, CalledProcessError


###########################################################################
# Change as per the need
###########################################################################
SEARCH_KEYWORD = "dog barking"
PATH_TO_WRITE_AUDIO = "youtube_scraped_audio/"
NUMBER_AUDIOCLIPS_LIMIT = 10


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
QUERY_CALL = SEARCH_KEYWORD
BASE_QUERY = "https://www.youtube.com/results?search_query="
QSTRING = "+".join(QUERY_CALL.split(" "))
REQUESTS_CALL = requests.get(BASE_QUERY + QSTRING)
PAGE = REQUESTS_CALL.text
SOUP = bs(PAGE, 'html.parser')
VIDEOS = SOUP.findAll('a', attrs={'class': 'yt-uix-tile-link'})


###########################################################################
# Create a list of query's with videoID's
###########################################################################
VIDEO_LIST = []
for v in VIDEOS:
    tmp = 'https://www.youtube.com' + v['href']
    VIDEO_LIST.append(tmp)


###########################################################################
# Download audio file onto the target directory
###########################################################################
COUNT = 0
for item in VIDEO_LIST[:NUMBER_AUDIOCLIPS_LIMIT]:
    COUNT += 1
    path = PATH_TO_WRITE_AUDIO + "_".join(SEARCH_KEYWORD.split(" ")) + "/" + \
           SEARCH_KEYWORD + "_" + item.split("=")[-1] + "_" + str(COUNT)
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
