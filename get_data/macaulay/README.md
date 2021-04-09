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

3. Once this file has executed run `macaulay_download` with the relevant arguments in the terminal.
   The relevant arguments can be viewed using `python macaulay_download.py -h`
   
  