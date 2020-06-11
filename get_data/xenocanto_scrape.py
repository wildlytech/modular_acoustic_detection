# bird sounds scrapping - audio data

from __future__ import unicode_literals
from bs4 import BeautifulSoup as bs
import requests
import youtube_dl
import argparse
import csv
import os
import pandas as pd


base_link = 'https://www.xeno-canto.org/explore?query='

DESCRIPTION = 'Scrape XenoCanto'
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
RequiredArguments = PARSER.add_argument_group('required arguments')
RequiredArguments.add_argument('-bird_species', action='store', \
    help='Input bird species by separating name with space and enclosed within quotes, \
    for instance "ashy prinia" ')
RequiredArguments.add_argument('-path_to_save_audio_files', action='store', \
    help='Input path to save audio files')
RESULT = PARSER.parse_args()

bird_species_keyword = RESULT.bird_species
audio_files_path = RESULT.path_to_save_audio_files

print "\nBird species keyword:", bird_species_keyword
print "Audio files path:", audio_files_path, '\n'

bird_species = bird_species_keyword.lower()
# replace whitespace with underscore for bird_species name
bird_species_name_ws = bird_species.replace(' ', '_')

# csv file name appended with bird species
csv_filename = audio_files_path+bird_species_name_ws+'/'+"xenocanto_bird_"+bird_species_name_ws+".csv"

print "csv file path:", csv_filename
#if not exists create directory with bird species name to save audio files
if not os.path.exists(audio_files_path+bird_species_name_ws):
    os.mkdir(audio_files_path+bird_species_name_ws)
else:
    pass

# Make a GET request to fetch the raw HTML content
r1 = requests.get(base_link+bird_species)
page1 = r1.text
soup1 = bs(page1, 'html.parser')

########################################################################################

# get number of pages in xento-canto for given bird species

result_pages = soup1.findAll('nav', attrs={'class':'results-pages'})
number_of_webpages = []
for result_page in result_pages:
    pages = result_page.find_all('li')
    for page1 in pages:
        number_of_webpages.append(page1.text.replace('\n', ' ').strip().encode('ascii'))
last_page = number_of_webpages[-2]
print "\nLast page for given Bird species:", last_page

###################################################################################################

def get_info_from_raw_html(page_number):
    # Make a GET request to fetch the raw HTML content
    r = requests.get(base_link+bird_species+'&pg='+str(page_number))
    print "\nPage link:", base_link+bird_species+'&pg='+str(page_number)
    page = r.text
    soup = bs(page, 'html.parser')

    # get audio file ID
    sub_url_list = soup.findAll('a', attrs={'class':'fancybox'})
    url_list = []
    # using id generate links to download audio files
    for v in sub_url_list:
        url = 'https://www.xeno-canto.org/' + v['title'].split(":")[0][2:]+ '/download'
        url_list.append(url)
    print "No of audio links in this page:", len(url_list), "\n"

    # get each row data
    row_data = soup.find_all('tr')
    # get only audio file information by removing column names row and
    # other website details row
    row_data_list = row_data[2:-1]
    return row_data_list

def get_rows_info(row_data):
    # get audio ID and add each row information to a list
    td_rows = []
    td = row_data.find_all('td')

    for each_row in td:
        # get audio id
        audio_info = row_data.find('a', attrs={'class':'fancybox'})
        audio_id = [audio_info['title'].split(":")[0].encode('ascii', 'ignore')]
        # remove any newlines and extra spaces from left and right
        td_rows.append(each_row.text.replace('\n', ' ').encode('ascii', 'ignore').strip())
    return td_rows, audio_id

def download_xc_audio(xc_audio_ID):
    # download xc audio file in the given path
    audio_link = 'https://www.xeno-canto.org/' + audio_ID[0][2:]+ '/download'
    print audio_link
    ydl_opts = {'outtmpl': audio_files_path+bird_species_name_ws+'/'+xc_audio_ID[0]+'.%(ext)s'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([audio_link])

###################################################################################################

column_tags = ['XenoCanto_ID', 'Common name/Scientific', 'Length', 'Recordist', 'Date', \
'Time', 'Country', 'Location', 'Elev(m)', 'Type', 'Remarks']

csv_file_exists = os.path.exists(csv_filename)
file_permission = 'a' if csv_file_exists else 'w'

# writing audio file information to csv
with open(csv_filename, file_permission) as csvfile:
    csvwriter = csv.writer(csvfile)
    if csv_file_exists:
        xc_csv = pd.read_csv(csv_filename, error_bad_lines=False)
        xc_id_in_csv = xc_csv["XenoCanto_ID"].values.tolist()
    else:
        csvwriter.writerow(column_tags)

    # iterate through all the pages
    for i in range(1, int(last_page)+1):
        row_data_lists = get_info_from_raw_html(i)

        for row1 in row_data_lists:
            rows_info, audio_ID = get_rows_info(row1)
            # check if csv file exists and duplication of audio info in csv file
            if csv_file_exists and (audio_ID[0] in xc_id_in_csv):
                continue
            else:
                csvwriter.writerow(audio_ID+rows_info[1:])

            # download the audio file
            download_xc_audio(audio_ID)

print "\nDone..!\n"
