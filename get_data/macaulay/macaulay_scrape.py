# Import relevant libraries
import os

import pandas
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.keys import Keys
import youtube_dl
import time
import argparse


class BrowserArgException(Exception):
    '''This exception thrown when invalid browser arg is passed'''
    pass


class BrowserPathException(Exception):
    '''This exception thrown when browser path does not exist'''
    pass


def get_ff_driver():
    '''
    This function returns a firefox webdriver. Make sure the ff driver
    is added to path
    '''
    options = FirefoxOptions()
    # Set browser in headless mode
    options.headless = True
    browser = webdriver.Firefox(options=options)

    return browser


def get_chrome_driver(chrome_path):
    '''
    The function returns a chrome web driver
    '''
    options = ChromeOptions()
    # Set browser in headless mode
    options.headless = True
    browser = webdriver.Chrome(chrome_path, options=options)
    return browser


def scrape(browser, query):
    '''
    This function involves the entire scraping functionality
    and saves the results as a dataframe with clipnames and asset ids
    which can be used to download the sounds.

    args: browser: The webdriver
    '''

    # Get the home page
    browser.get(args.link)

    # Get the text input element
    element = browser.find_element_by_id("hero-search")
    # Input the bird name as a query
    element.send_keys(query)
    # Sleep to press enter
    print("Query just entered in the browser, waiting for suggestions to load..")
    time.sleep(3)
    # Press enter to make the query
    element.send_keys(Keys.RETURN)
    print("Enter key just pressed, waiting for results to load...")
    time.sleep(5)
    # Get the current url
    url = browser.current_url
    # Create the insertion url (for mediatype = audio)
    insertion = "&mediaType=a&"
    # Split url to add insertion
    split_url = url.split("&")
    # New url
    new_url = split_url[0] + insertion + split_url[1]
    # Make the browser go to the new url
    browser.get(new_url)
    while (True):
        try:
            print("Show more button found...")
            button = browser.find_element_by_id("show_more")
            button.click()
        except:
            print("Record Limit Reached..")
            break
    # Get the elements for getting asset ids
    results = browser.find_elements_by_class_name("ResultsGallery-link")

    # Initialise relevant variables
    asset_id_list = []
    bird_name = query + "_macaul_"
    k = 1
    bird_names = []

    # Iterate through results to fetch asset ids and store them in lists

    for row in results:
        asset_id = row.get_attribute("data-asset-id")

        asset_id_list.append(asset_id)
        bird_names.append(bird_name + str(k))
        k += 1

    # Create dataframe and save it
    print("Creating Bird DF...")
    print(len(set(asset_id_list)))
    # asset_id_list = list(set(asset_id_list))
    # bird_names = bird_names[:len(asset_id_list)]
    df = pandas.DataFrame({"ClipName": bird_names, "Asset_ID": asset_id_list})
    print(df.Asset_ID.value_counts())
    print("Number of results: ", len(df))
    return df
    # Close the browser
    browser.close()


def download_clip(audio_id, clip_name, save_path,
                  root_link_path="https://cdn.download.ams.birds.cornell.edu/api/v1/asset/"):
    id_1 = audio_id
    audio_link = root_link_path + str(id_1)
    audio_files_path = save_path + clip_name
    ydl_opts = {'outtmpl': audio_files_path + '.mp3'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([audio_link])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scrapes macaulay audio lib using selenium")
    parser.add_argument("-b", "--browser", help="Browser of choice: C for Chrome and F for firefox", required=True)
    parser.add_argument("-sp", "--save_path", help="Path to save the audio clips", default="macaulay_audio/")
    parser.add_argument("-cp", "--chrome_path", help="If chrome is the browser of choice, path to its webdriver")
    parser.add_argument("-l", "--link", help="Link to web page to be scraped",
                        default="https://www.macaulaylibrary.org/")
    parser.add_argument("-q", "--query_name", help="Name of bird whose audio is required")
    parser.add_argument("-qp", "--query_path", help="Path to text file with required bird names")

    args = parser.parse_args()

    driver = args.browser

    if args.query_path:
        print("Scraping using path to query text file")
        with open(args.query_path, "r") as f:
            queries = f.readlines()
        queries = [query.rstrip("\n") for query in queries]
        df_list = []
        for query in queries:
            print("Scraping for bird ", query)
            try:
                if driver == "C" or driver == "c":
                    try:
                        if os.path.exists(args.chrome_path):
                            browser = get_chrome_driver(args.chrome_path)
                        else:
                            raise BrowserPathException
                    except BrowserPathException:
                        print("Chrome Driver path does not exist")
                elif driver == "F" or driver == "f":
                    try:
                        browser = get_ff_driver()
                    except BrowserPathException:
                        print("Gecko driver path not added to PATH. Try moving driver to /usr/local/bin")

                else:
                    raise BrowserArgException

                df = scrape(browser, query)
                df_list.append(df)

            except BrowserPathException as e:
                print(e)

        try:
            final_df = pandas.concat(df_list)
            final_df.to_csv(args.save_path)
            print("DF Saved to: ", args.save_path)
            for i in range(len(final_df)):
                audio_id = final_df["Asset_ID"][i]
                clip_name = "Macaulay_" + str(audio_id)
                download_clip(audio_id, clip_name, args.save_path)
        except Exception:
            print("Something went wrong while making the final Dataframe")

    else:
        print("Scraping using query name")
        try:
            if driver == "C" or driver == "c":
                try:
                    if os.path.exists(args.chrome_path):
                        browser = get_chrome_driver(args.chrome_path)
                    else:
                        raise BrowserPathException
                except BrowserPathException:
                    print("Chrome Driver path does not exist")
            elif driver == "F" or driver == "f":
                try:
                    browser = get_ff_driver()
                except BrowserPathException:
                    print("Gecko driver path not added to PATH. Try moving driver to /usr/local/bin")

            else:
                raise BrowserArgException

            df = scrape(browser, args.query_name)
            for i in range(len(df)):
                audio_id = df["Asset_ID"][i]
                clip_name = df["ClipName"][i] + "_" + str(audio_id)
                download_clip(audio_id, clip_name, args.save_path)

            # print("Creating and saving df...")
            # df.to_csv(args.save_path)

        except BrowserPathException as e:
            print(e)
