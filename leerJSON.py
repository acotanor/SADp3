import json
import sys

# Dado un archivo JSON devuelve un diccionario con el formato:
# atributo : valor
def leerJson(file):
    try:
        with open(file,"r") as jsonFile:    # Abre el archivo.
            return json.load(jsonFile)      # Carga los datos en un diccionario.
    except FileNotFoundError:
        return Null
    except json.JSONDecodeError:
        return Null