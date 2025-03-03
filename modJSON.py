import json
import sys
import leerJSON

# Dado un archivo JSON, el atributo a cambiar y el valor modificado, modifica los metadatos.
def modJson(file, atributo, valor):
    data = leerJSON.leerJson(file)              # Lee el archivo JSON a modificar.
    data["preprocessing"][atributo] = valor     # Cambiamos el valor de un atributo.

    with open(file, "w") as jsonFile:        
        json.dump(data, jsonFile, indent=4)     # Aplicamos los cambios en el archivo.
        
    print(leerJSON.leerJson(file))
