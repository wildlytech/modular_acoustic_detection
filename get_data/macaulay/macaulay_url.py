import requests
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser(description="Scrapes bird names from argument checklist")
parser.add_argument("-u", "--url", help="URL for checklist", required=True)
parser.add_argument("-sp", "--save_path", help="Path to save file", required=True)

args = parser.parse_args()
url = args.url
save_path = args.save_path
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")

name_elements = [element.text for element in soup.find_all("span", class_="Heading-main")]
data_species_code = [element["data-species-code"] for element in soup.find_all("a",attrs={"data-species-code":True})]

s = ""
for name in name_elements[1:]:
    s += name + "\n"


with open(save_path, "w+") as f:
    f.write(s)

s=""
for code in data_species_code:
    s+= code+"\n"

save_path = "bird-codes.txt"
with open(save_path, "w+") as f:
    f.write(s)
