import requests
from PIL import Image
from bs4 import BeautifulSoup
import pandas as pd
import os
import json
import time
import random
import shutil

# Create directories if they don't exist
img_folder_path = "/data/LAYER0/MO/MO"

if not os.path.exists(img_folder_path):
    os.makedirs(img_folder_path)

# Load species and image IDs from CSV
csv_file_path = '../../notebooks/mushroom_observer/dataset-mushroom-observer/qualified-dataset/all_ids.csv'
df = pd.read_csv(csv_file_path)
#test
df = df.head(1)
print(df.head())
# Create a dictionary of species and their corresponding image IDs
species_dict = {}
for index, row in df.iterrows():
    overall_id = str(row["Overall_ID"])
    merged_ids = row["Merged_IDs"].split(',')
    species_dict[overall_id] = merged_ids

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
base_url = 'https://mushroomobserver.org/'
session = requests.Session()
session.headers.update(headers)

# Download images based on JSON files
json_folder = '../../notebooks/mushroom_observer/dataset-mushroom-observer/qualified-dataset'
json_files = os.listdir(json_folder)
for json_file in json_files:
    if not json_file.startswith('specie-id-'):
        continue

    specie_id = json_file[10:json_file.index('.json')]
    specie_folder = os.path.join(img_folder_path, specie_id)
    if not os.path.isdir(specie_folder):
        os.makedirs(specie_folder)

    with open(os.path.join(json_folder, json_file), "r") as f:
        data = json.loads(f.read())
        images_ids = data["upSelection"] + data["downSelection"]

        for image_id in images_ids:
            image_path = os.path.join(specie_folder, f"{image_id}.jpg")
            if not os.path.isfile(image_path):
                response = requests.get(f"https://images.mushroomobserver.org/960/{image_id}.jpg") # 960 = image resolution
                if response.status_code == 200:
                    with open(image_path, "wb") as img_file:
                        img_file.write(response.content)

# Scrape additional images and metadata from the web pages
data = []
for index, row in df.iterrows():
    species_id = row['Overall_ID']
    image_ids = row['Merged_IDs'].split(',')

    for image_id in image_ids:
        page_url = f'{base_url}{image_id}'
        url_reference = image_id

        sleep_time = random.randint(5, 15)
        time.sleep(sleep_time)

        try:
            response = session.get(page_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            name_tag = soup.find('h1', id='title').find('i')
            name = name_tag.get_text(strip=True) if name_tag else f'Unknown_{image_id}'

            img_tags = soup.find_all('img', class_='carousel-image')
            img_counter = 1
            for img_tag in img_tags:
                img_url = img_tag.get('data-src')
                if img_url:
                    img_data = session.get(img_url).content
                    img_name = f"{url_reference}_{name.replace(' ', '_')}_{img_counter}.jpg"
                    img_path = os.path.join(img_folder_path, img_name)

                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_data)

                    data.append({'URL': page_url, 'Name': name, 'Image': img_name})
                    img_counter += 1

        except requests.exceptions.RequestException as e:
            print(f"Error during request {page_url}: {e}")
            continue

df = pd.DataFrame(data)
output_path = 'data/LAYER0/MO/dataset.csv'
df.to_csv(output_path, index=False)

print(f'Data saved to {output_path}')