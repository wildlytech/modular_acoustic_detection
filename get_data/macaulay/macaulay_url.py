import requests
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser(description="Scrapes bird codes from argument checklist")
parser.add_argument("-u", "--url", help="URL for checklist", required=True)
parser.add_argument("-sp", "--save_path", help="Path to save file", required=True,default="bird-codes.txt")

args = parser.parse_args()
url = args.url
save_path = args.save_path
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")

name_elements = [element.text for element in soup.find_all("span", class_="Heading-main")]
data_species_code = [element["data-species-code"] for element in soup.find_all("a", attrs={"data-species-code": True})]


s = ""
for code in data_species_code:
    s += code + "\n"

with open(save_path, "w+") as f:
    f.write(s)
