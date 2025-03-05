import json
import sys

# Dado un archivo JSON devuelve un diccionario con el formato:
# atributo : valor
def leerJson(file) -> dict:
    try:
        with open(file,"r") as jsonFile:        # Abre el archivo.
            return json.load(jsonFile)          # Carga los datos en un diccionario.
    except FileNotFoundError:
        print(f"El archivo {file} no existe.")
        return {}                               # Diccionario vacío.
    except json.JSONDecodeError:
        print("Formato incorrecto.")
        return {}
    except Exception as e:
        print(f"Error: {e}")
        return {}