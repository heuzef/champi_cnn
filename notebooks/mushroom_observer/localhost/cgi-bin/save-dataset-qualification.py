#!/usr/bin/env python
# coding: utf-8

import cgi 
import json
import os

form = cgi.FieldStorage()

data = json.loads(form.getvalue("jsonData"))

if os.path.isfile("notebooks/mushroom_observer/dataset-mushroom-observer/qualified-dataset/specie-id-"+form.getvalue("specieId")+".json"):
    with open("notebooks/mushroom_observer/dataset-mushroom-observer/qualified-dataset/specie-id-"+form.getvalue("specieId")+".json", "r") as f:
        previousData = json.loads(f.read())
        data["upSelection"] = data["upSelection"] + previousData["upSelection"]
        data["downSelection"] = data["downSelection"] + previousData["downSelection"]
        data["exclusion"] = data["exclusion"] + previousData["exclusion"]


with open("notebooks/mushroom_observer/dataset-mushroom-observer/qualified-dataset/specie-id-"+form.getvalue("specieId")+".json", "w") as f:
    f.write(json.dumps(data))