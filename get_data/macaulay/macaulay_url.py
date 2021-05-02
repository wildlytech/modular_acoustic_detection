import requests
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser(description="Scrapes bird codes from argument checklist")
parser.add_argument("-u", "--url", help="URL for checklist", default="https://ebird.org/region/IN/media?yr=all&m=")
parser.add_argument("-csp", "--save_path_codes", help="Path to save bird codes", default="bird-codes.txt")
parser.add_argument("-nsp", "--save_path_names", help="Path to save bird names", default="bird-names.txt")

args = parser.parse_args()
url = args.url
save_path_codes = args.save_path_codes
save_path_names = args.save_path_names
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")

print("Scraping...")
name_elements = [element.text for element in soup.find_all("span", class_="Heading-main")]
data_species_code = [element["data-species-code"] for element in soup.find_all("a", attrs={"data-species-code": True})]

s = ""

for name in name_elements:
    s += name + "\n"

print("Saving bird names..")
with open(save_path_names, "w+") as f:
    f.write(s)

s = ""
for code in data_species_code:
    s += code + "\n"

print("Saving bird codes...")
with open(save_path_codes, "w+") as f:
    f.write(s)
