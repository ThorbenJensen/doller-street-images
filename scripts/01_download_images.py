"""Download images."""

import json
import re
import time

import requests
from bs4 import BeautifulSoup
from toolz import pipe
from toolz.curried import map

# %% read data
PATH = "data/beds.html"

with open(PATH, 'r') as fp:
    data = fp.read().replace('\n', '')

# %% apply beautifulsoup
soup = BeautifulSoup(data, "html.parser")

matrix = soup.find("matrix-images").find_all("div", {"class": "cell-inner"})

image_links = pipe(matrix,
                   map(lambda x: x['style']),
                   map(lambda x: re.findall("url\\(\"(.+)\"\\);", x)[0]),
                   map(lambda x: "http:" + x),
                   list)

image_ids = pipe(image_links,
                 map(lambda x: re.findall("image/([a-z0-9\\-]+)/", x)[0]),
                 list)

income = pipe(matrix,
              map(lambda x: x.find('span').contents[0]),
              map(lambda x: re.sub("[$ ]", "", x)),
              map(int),
              list)

assert len(matrix) == len(image_links) == len(image_ids) == len(income), \
    "Length of parsed lists differs!"


# %% download images
def download_image(url: str, out_name: str, out_folder: str = "data/images/"):
    response = requests.get(url)
    if response.status_code == 200:
        with open(out_folder + out_name, 'wb') as f:
            f.write(response.content)


for image_link, image_id in zip(image_links, image_ids):
    print(f"Downloading {image_id}...")
    download_image(url=image_link, out_name=image_id + ".jpg")
    time.sleep(1)

# %% save label (income)
labels = dict(zip(image_ids, income))

with open('data/labels.json', 'w') as fp:
    json.dump(labels, fp, indent=4)
