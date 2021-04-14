# Macaulay Scraper
## Info
This directory consists of two files

1. `macaulay_scraper.py`: This file scrapes the asset ids from the macaulay website and downloads the audio files
2. `macaulay_url.py`: This file downloads the names of birds from a given url

## Overview of run
In order to automate the process of scraping, we use something called selenium which
automates a browser and retrieves the audio files. It is equivalent to someone using the browser
manually. In order to automate the browser we need programs called webdrivers. They act 
like a bridge between the python script and the browser. There are different webdrivers
for firefox and chrome

##Steps to run
1. Download the relevant webdriver for your system:
   <table>
   <tr>
   <td>Driver</td>
   <td>Link</td>
   </tr>
   <tr>
   <td>Chrome</td>
   <td><a href="https://chromedriver.chromium.org/downloads">https://chromedriver.chromium.org/downloads</a></td>
   </tr>
   <tr>
   <td>Firefox</td>
   <td><a href="https://github.com/mozilla/geckodriver/releases">https://github.com/mozilla/geckodriver/releases</a> </td>
   </tr>
   </table>
   Other webdrivers can be found <a href="https://selenium-python.readthedocs.io/installation.html">here</a><br>
   If using the firefox webdriver make sure to save it to the /usr/bin folder (Ubuntu or Mac)
2. Run `macaulay_scrape.py` with the relevant arguments in the terminal. The relevant arguments can be viewed using
   `python macaulay_scrape.py -h`

    The arguments are as follows:
    - `-b` : The browser argument. This selects the browser the script will use to scrape 
    the website. The value can be either `C` for chrome or `F` for firefox.
    - `-sp` : The save path argument. This is the path to the folder the script will
    save the bird audio files on script completion.
    - `-cp` : The chrome path argument. It is only applicable is using the chrome webdriver.
    It is the path to where the chrome webdriver is saved. You can leave this argument
    if using the firefox webdriver.
    - `-l` : The link argument. This is the link to the webpage you want to scrape. It has
    a default value of the macaulay website. Can be ignored altogether.
    - `-q` : The query argument. This is the name of the bird whose audio you want to scrape.
    Enter this if not entering the `-qp` argument.
    - `-qp` : The query path argument. This is the name of the file created by the `macaulay_url.py`
    file or can be created by hand. It is a text file which contains names of birds at every line.
    Ignore this if entering `-q` argument.
3. While using `macaulay_scrape.py` if using chrome driver, pass the path to the
chromedriver as an argument. Else if using firefox webdriver, put the firefox webdriver in the `/usr/bin/` directory before running the script.
Do not pass path to the firefox driver while running the script with firefox.
   
4. Run `macaulay_url.py` to get the bird names for a particular url checklist
and save it to a text file. 
    The arguments for the same are:
    - `-u`: The url for the checklist which will contain the names of birds of a particular region.
    - `-sp`: The path to the file where you want to save the bird names.
    
