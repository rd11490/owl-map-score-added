import requests
import re
import os
import zipfile

# Make a request to get the html of the overwatch league stats lab page
resp_text = requests.get("https://overwatchleague.com/en-us/statslab").text
# Regex extract all zip file hrefs on the page
links = re.findall( r'(https://assets.*?.zip)', resp_text)

zip_dir_name = 'zips'
map_data_dir_name = 'map_data'
player_data_dir_name = 'player_data'


# Create a zips directory if it doesn't exist
if not os.path.isdir(zip_dir_name):
    os.mkdir(zip_dir_name)

# Create a data directory if it doesn't exist
if not os.path.isdir(map_data_dir_name):
    os.mkdir(map_data_dir_name)


for l in links:
    # Pull the zip name from the href
    zip_name = l.split('/')[-1]

    # Make a request to get the zip file
    r = requests.get(l, stream=True)

    # save the zip file into the zips folder
    with open('{}/{}'.format(zip_dir_name, zip_name), 'wb') as fd:
        for chunk in r.iter_content(chunk_size=512):
            fd.write(chunk)

zips = os.listdir(zip_dir_name)

for z in zips:
    with zipfile.ZipFile('{}/{}'.format(zip_dir_name, z), 'r') as zip_ref:
        if 'map' in z:
            zip_ref.extractall(map_data_dir_name)
        else:
            zip_ref.extractall(player_data_dir_name)


