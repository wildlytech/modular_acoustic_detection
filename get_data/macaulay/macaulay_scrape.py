# Import relevant libraries
import os

import pandas
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.keys import Keys

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
    browser = webdriver.Firefox()
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


def scrape(browser):
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
    element.send_keys(args.query_name)
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
    # Get the elements for getting asset ids
    results = browser.find_elements_by_class_name("ResultsGallery-link")

    # Initialise relevant variables
    asset_id_list = []
    bird_name = args.query_name + "_macaul_"
    k = 1
    bird_names = []

    # Iterate through results to fetch asset ids and store them in lists

    for row in results:
        asset_id = row.get_attribute("data-asset-id")
        asset_id_list.append(asset_id)
        bird_names.append(bird_name + str(k))
        k += 1

    # Create dataframe and save it
    print("Creating and saving DF...")
    df = pandas.DataFrame({"ClipName": bird_names, "Asset_ID": asset_id_list})
    df.to_csv("Asset_ids.csv")
    print("DF Saved...")
    # Close the browser
    browser.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scrapes macaulay audio lib using selenium")
    parser.add_argument("-b", "--browser", help="Browser of choice: C for Chrome and F for firefox", required=True)
    parser.add_argument("-q", "--query_name", help="Name of bird whose audio is required", required=True)
    parser.add_argument("-cp", "--chrome_path", help="If chrome is the browser of choice, path to its webdriver")
    parser.add_argument("-l", "--link", help="Link to web page to be scraped",
                        default="https://www.macaulaylibrary.org/")
    args = parser.parse_args()

    driver = args.browser
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

        scrape(browser)
    except BrowserPathException as e:
        print(e)
