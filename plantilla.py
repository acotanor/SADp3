import pandas as pd
import sys

def load_data(file):
    # Carga de datos
    data = pd.read_csv(file)
    return data

def select_features(data, columna):
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (list): Lista que contiene las características numéricas.
        text_feature (list): Lista que contiene las características de texto.
        categorical_feature (list): Lista que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']).columns.tolist() # Columnas numéricas
        if columna in numerical_feature:
            numerical_feature.remove(columna)
        
        # Categorical features
        categorical_feature = data.select_dtypes(include='object').columns.tolist()
        
        # Text features
        text_feature = [col for col in data.select_dtypes(include='object').columns if col not in categorical_feature]
        
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print("Error al separar los datos")
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    # Carga de datos
    file_path = input("Introduce la ruta del archivo: ")
    data = load_data(file_path)
    
    # Seleccionar características
    numerical_feature, text_feature, categorical_feature = select_features(data, "Especie")
    
    # Mostrar las características
    print("Características numéricas:", numerical_feature)
    print("Características de texto:", text_feature)
    print("Características categóricas:", categorical_feature)