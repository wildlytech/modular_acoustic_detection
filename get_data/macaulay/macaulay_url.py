import requests
from bs4 import BeautifulSoup

url = "https://ebird.org/region/IN/media?yr=all&m="

page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")

name_elements = [element.text for element in soup.find_all("span", class_="Heading-main")]

s = ""
for name in name_elements[1:]:
    s += name + "\n"

with open("macaulay_birds.txt", "w+") as f:
    f.write(s)
