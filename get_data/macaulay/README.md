# Macaulay Scraper
## Info
This directory consists of two files

1. `macaulay_scraper.py`: This file scrapes the asset ids from the macaulay website and stores them in a csv
2. `macaulay_download.py`: This file downloads the asset ids using the scraped by the first file.

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
   Other webdrivers can be found <a href="https://selenium-python.readthedocs.io/installation.html">here</a>
   
2. Run `macaulay_scrape.py` with the relevant arguments in the terminal. The relevant arguments can be viewed using
   `python macaulay_scrape.py -h`

3. While using `macaulay_scrape.py` if using chrome driver, pass the path to the
chromedriver as an argument. Else if using firefox webdriver, put the firefox webdriver in the `/usr/bin/` directory before running the script.
Do not pass path to the firefox driver while running the script with firefox.

4. Once this file has executed run `macaulay_download` with the relevant arguments in the terminal.
   The relevant arguments can be viewed using `python macaulay_download.py -h`
   
5. Run `macaulay_url.py` to get the bird names for a particular url checklist
and save it to a text file.