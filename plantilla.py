import pandas
import sys

def load_data(file):
    # Carga de datos
    data = pandas.read_csv(file)
    return data
if __name__ == '__main__':
    # Carga de datos
    file_path = input("Introduce la ruta del archivo: ")
    data = load_data(file_path)
    print(data.head())