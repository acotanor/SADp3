import json
import sys
import leerJSON

# Dado un archivo JSON, el atributo a cambiar y el valor modificado, modifica los metadatos.
def modJson(file, atributo, valor):
    try:
        data = leerJSON.leerJson(file)              # Lee el archivo JSON a modificar.
        data["preprocessing"][atributo] = valor     # Cambiamos el valor de un atributo.

        with open(file, "w") as jsonFile:        
            json.dump(data, jsonFile, indent=4)     # Aplicamos los cambios en el archivo.
    except FileNotFoundError:
        print(f"El archivo {file} no existe.")
    except json.JSONDecodeError:
        print("Formato incorrecto.")
    except Exception as e:
        print(f"Error: {e}")