import json
import sys

def leerJson(file):
    with open(file,"r") as jsonFile:
        data = json.load(jsonFile)
    print(data)


leerJson("ej.json")