import pandas as pd
import numpy as np
import sys
import os
import argparse
import pickle
import json
import csv
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
#NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

#    ____                                    _             _        ____        _            
#   |  _ \ _ __ ___   ___ ___  ___  __ _  __| | ___     __| | ___  |  _ \  __ _| |_ ___  ___ 
#   | |_) | '__/ _ \ / __/ _ \/ __|/ _` |/ _` |/ _ \   / _` |/ _ \ | | | |/ _` | __/ _ \/ __|
#   |  __/| | | (_) | (_|  __/\__ \ (_| | (_| | (_) | | (_| |  __/ | |_| | (_| | || (_) \__ \
#   |_|   |_|  \___/ \___\___||___/\__,_|\__,_|\___/   \__,_|\___| |____/ \__,_|\__\___/|___/

def generarJson(filename="config.json"):
    """
    Comprueba si existe un archivo de configuración, de no haberlo crea uno por defecto.

    Args:
        filename: El nombre del archivo, siempre usamos el mismo por lo que tiene un valor default.

    Returns:
        None

    Raises:
        None
    """
    if not os.path.exists(filename):
        print(f"Generando un archivo de configuración: {filename}...")
        
        config = {
            "preprocessing": {
                "categorical_features": ["A", "B", "C"],
                "missing_values": "D",
                "impute_strategy": "mean",
                "scaling": "standard",
                "text_process": "tf_idf",
                "sampling": "undersampling"
            }
        }

        with open(filename, "w") as jsonFile:
            json.dump(config, jsonFile, indent=4)
        
        print(f"Archivo de configuración {filename} creado correctamente.")
    else:
        return

def leerJson(file) -> dict:
    """
    Dado un archivo JSON devuelve un diccionario con sus metadatos.

    Args:
        file: El archivo .json
    
    Returns:
        dict: Un diccionario con los metadatos del archivo .json
    
    Raises:
        FileNotFoundError:  No existe el archivo .json especificado, devuelve un diccionario vacío.
        JSONDecodeError:    El formato del archivo .json es incorrecto y no se puede decodificar, devuelve un diccionario vacío.
        Exception:          Ha habido un error inesperado, devuelve un diccionario vacío. 
    """
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

def modJson(file, atributo, valor):
    """
    Dado un archivo JSON, el atributo a cambiar y el valor modificado, modifica los metadatos.

    Args:
        file: El archivo .json.
        atributo: El atributo a cambiar.
        valor: El valor nuevo del atributo.
    
    Returns:
        None
    
    Raises:
        FileNotFoundError:  No existe el archivo .json especificado, devuelve un diccionario vacío.
        JSONDecodeError:    El formato del archivo .json es incorrecto y no se puede decodificar, devuelve un diccionario vacío.
        Exception:          Ha habido un error inesperado, devuelve un diccionario vacío. 
    """
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

def load_data(file, encoding='utf-8'):
    # Carga de datos
    try:
        data = pd.read_csv(file, encoding=encoding)
    except UnicodeDecodeError:
        print(f"Error al decodificar el archivo con la codificación {encoding}. Intentando con 'latin1'.")
        data = pd.read_csv(file, encoding='latin1')
    return data

def select_features(data):
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (list): Lista que contiene las características numéricas.
        text_feature (list): Lista que contiene las características de texto.
        categorical_feature (list): Lista que contiene las características categóricas.
    """
    try:
        # Eliminar columnas no deseadas
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if args.column in numerical_feature:
            numerical_feature.remove(args.column)
        
        # Categorical features
        categorical_feature = data.select_dtypes(include='object').columns.tolist()
        if args.column in categorical_feature:
            categorical_feature.remove(args.column)
        
        # Text features
        text_feature = [col for col in categorical_feature if col != 'v1']
        if args.column in text_feature:
            text_feature.remove(args.column)
        
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print("Error al separar los datos")
        print(e)
        sys.exit(1)

def process_missing_values(data, numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        data (DataFrame): El DataFrame que contiene los datos originales.
        numerical_feature (list): Lista que contiene las características numéricas.
        categorical_feature (list): Lista que contiene las características categóricas.

    Returns:
        DataFrame: El DataFrame con los valores faltantes procesados.

    Raises:
        None
    """
    try:
        # Procesar valores faltantes en características numéricas
        for feature in numerical_feature:
            data[feature] = data[feature].fillna(data[feature].mean())  # Rellenar con la media

        # Procesar valores faltantes en características categóricas
        for feature in categorical_feature:
            data[feature] = data[feature].fillna(data[feature].mode()[0])  # Rellenar con la moda

        return data
    except Exception as e:
        print("Error al procesar los valores faltantes")
        print(e)
        sys.exit(1)

def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.

    Returns:
        numerical_feature: El dato de entrada reescalado.

    Raises:
        Exception: Si hay un error al reescalar los datos.

    """
    scaler = MinMaxScaler(feature_range=(0,1))
    return scaler.fit_transform(numerical_feature)


def cat2num(data, categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Args:
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    Returns:
        categorical_feature: El dato de entrada convertido en valor numérico.
    """
    le = LabelEncoder()
    
    for feature in categorical_feature:
        data[feature] = le.fit_transform(data[feature])
    
    return data

def simplify_text(data, text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower, stemmer, tokenizer, stopwords del NLTK....
    
    Parámetros:
    - data: DataFrame - El DataFrame que contiene la columna de texto a simplificar.
    - text_feature: list - Lista de nombres de las columnas de texto a simplificar.
    
    Retorna:
    DataFrame: El DataFrame con las columnas de texto simplificadas.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for feature in text_feature:
        # Convertir a minúsculas
        data[feature] = data[feature].str.lower()
        
        # Tokenizar
        data[feature] = data[feature].apply(word_tokenize)
        
        # Eliminar stopwords y aplicar stemming
        data[feature] = data[feature].apply(lambda x: [stemmer.stem(word) for word in x if word not in stop_words])
        
        # Unir tokens de nuevo en una cadena
        data[feature] = data[feature].apply(lambda x: ' '.join(x))
    
    return data

def process_text(text_feature, text_process):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (list): Una lista que contiene los nombres de las características de texto a procesar.
    text_process (str): Técnica de procesamiento de texto a utilizar ("tf-idf" o "bow").

    """
    global data
    try:
        if len(text_feature) > 0:
            if text_process == "tf-idf":               
                tfidf_vectorizer = TfidfVectorizer()
                text_data = data[text_feature].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                data.drop(columns=text_feature, inplace=True)
                print("Texto tratado con éxito usando TF-IDF")
            elif text_process == "bow":
                bow_vectorizer = CountVectorizer()
                text_data = data[text_feature].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vectorizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                data.drop(columns=text_feature, inplace=True)
                print("Texto tratado con éxito usando BOW")
            else:
                print("No se están tratando los textos")
        else:
            print("No se han encontrado columnas de texto a procesar")
    except Exception as e:
        print("Error al tratar el texto")
        print(e)
        sys.exit(1)

def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.sampling
    Args:
        None
    
    Returns:
        None
    
    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """


def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """

def preprocesar_datos(text_process):
    global data
    numerical_feature, text_feature, categorical_feature = select_features(data)
    
    # Procesar valores faltantes
    data = process_missing_values(data, numerical_feature, categorical_feature)
    # Simplificar texto
    data = simplify_text(data, text_feature)
    # Procesar texto
    process_text(text_feature, text_process)
    

    # Convertir variables categóricas a numéricas
    data = cat2num(data, categorical_feature)
    print("Columnas del DataFrame después de categorizar:", data.columns)

#    __  __           _      _           
#   |  \/  | ___   __| | ___| | ___  ___ 
#   | |\/| |/ _ \ / _` |/ _ \ |/ _ \/ __|
#   | |  | | (_) | (_| |  __/ | (_) \__ \
#   |_|  |_|\___/ \__,_|\___|_|\___/|___/

def divide_data():
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Parámetros:
    None

    Returns:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo.
    """
    global data
    x = data.iloc[:, :-1].values # Todas las columnas menos la última
    y = data.iloc[:, -1].values # Última columna
    
    # Dividimos los datos en entrenamiento y test
    return train_test_split(x, y, test_size=args.test)

def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        with open(f'output/{args.modelo}.pkl', 'wb') as file:
            pickle.dump(gs, file)
            print("Modelo guardado con éxito")
        with open(f'output/{args.modelo}.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
    except Exception as e:
        print("Error al guardar el modelo")
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """

    print("> Mejores parametros:\n", gs.best_params_)
    print("> Mejor puntuacion:\n", gs.best_score_)
    print("> F1-score micro:\n", calculate_fscore(y_dev, gs.predict(x_dev))[0])
    print("> F1-score macro:\n", calculate_fscore(y_dev, gs.predict(x_dev))[1])
    print("> Informe de clasificación:\n", classification_report(y_dev, gs.predict(x_dev)))
    print("> Matriz de confusión:\n", confusion_matrix(y_dev, gs.predict(x_dev)))

def calculate_fscore(y_true, y_pred):
    """
    Calcula el F1-score de un clasificador. 
    """
    return f1_score(y_true, y_pred, average='micro'), f1_score(y_true, y_pred, average='macro')

def knn():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    x_train = reescaler(x_train)
    x_dev = reescaler(x_dev)
    
    # Hacemos un barrido de hiperparametros
    gs = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [1,2,3,4,5,6,7,8,9,10]}, n_jobs=args.cpu)
    gs.fit(x_train, y_train)

    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    """knn = KNeighborsClassifier(n_neighbors=8)
    y_pred = knn.predict(x_dev)"""
    # Guardamos el modelo utilizando pickle
    save_model(gs)


def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        #TODO Llamar al decision trees
        #gs = GridSearchCV(

        execution_time = end_time - start_time
    print("Tiempo de ejecución:"  + execution_time + "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)


def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:
        #TODO Llamar al decision trees
        #gs = GridSearchCV(
        execution_time = end_time - start_time
    
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open(f'output/{args.modelo}.pkl', 'rb') as file:
            model = pickle.load(file)
            print("Modelo cargado con éxito")
            return model
    except Exception as e:
        print("Error al cargar el modelo")
        print(e)
        sys.exit(1)

def predict(modelo):
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    # Predecimos
    prediction = modelo.predict(data)
    
    # Añadimos la prediccion al dataframe data:
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.column])], axis=1)
    # Guardamos el dataframe con la predicción:
    data.to_csv('output/data-prediction.csv',index=False)

#    __  __       _       
#   |  \/  | __ _(_)_ __  
#   | |\/| |/ _` | | '_ \ 
#   | |  | | (_| | | | | |
#   |_|  |_|\__,_|_|_| |_|
                    
if __name__ == '__main__':
    # Procesamiento de los argumentos de entrada
    parser = argparse.ArgumentParser(description="Procesamiento de datos")
    parser.add_argument('-d','--data', type=str, help="Ruta del archivo CSV", required=True)
    parser.add_argument('-m','--mode', type=str, choices=['train','test'], help="Modo de ejecución (train/test)", required=True)
    parser.add_argument('--text_process', type=str, choices=['tf-idf', 'bow'], help="Técnica de procesamiento de texto a utilizar", required=True)
    parser.add_argument('-s','--sampling', type=str, choices=['over','under'], help="Realizar over o under sampling a los datos en los que sea necesario.", required=True)
    parser.add_argument('-o','--output', type=str, help="Ruta del archivo CSV de salida", required=True)
    parser.add_argument('-c','--column', type=str, help="Columna a predecir.")
    parser.add_argument('--modelo', type=str, choices=["knn","dt","rf"], help="Modelo a entrenar/probar. (knn=KNN, dt=Decision Tree, rf=Random Forest)")
    parser.add_argument("--cpu", help="Número de CPUs a utilizar (-1 para usar todos)", default=-1, type=int)
    parser.add_argument("--test", type=float, default=0.25,help="La proporción datos de entrenamiento / datos de test.")

    args = parser.parse_args()

    np.random.seed(1)  # Utilizamos una semilla para poder reproducir los resultados.

    # Creamos la carpeta output:
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print("Carpeta output creada con éxito.")
    except FileExistsError:
        print("La carpeta output ya existe.")
    except Exception as e:
        print("Error al crear la carpeta output.")
        print(e)
        sys.exit(1)

    # Generamos la configuración:
    generarJson()

    # Carga de datos:
    print(f"\n- Cargando datos desde {args.data}...")
    data = load_data(args.data)

    # Descargamos recursos necesarios de NLTK
    print("\n- Descargando diccionarios...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    nltk.download('punkt_tab')

    # Preprocesar datos:
    preprocesar_datos(args.text_process)
        
    # Guardar datos procesados en un nuevo archivo CSV
    data.to_csv(args.output, index=False)
    print(f"Datos procesados guardados en {args.output}")

    if args.mode=='train':
        # Entrenamos el modelo seleccionado:
        print(f"Entrenando el modelo {args.modelo}...")
        if args.modelo=="knn":
            try:
                knn()
                print("Modelo KNN entrenado con éxito.")
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.modelo=="dt":
            try:
                decision_tree()
                print("Modelo Decision Tree entrenado con éxito.")
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.modelo=="rf":
            try:
                random_forest()
                print("Modelo Random Forest entrenado con éxito.")
                sys.exit(0)
            except Exception as e:
                print(e)
    elif args.mode=='test':
        print(f"\n- Cargando el modelo {args.modelo}...")
        modelo = load_model()
        try:
            predict(modelo)
            print(f"Test del modelo {args.modelo} realizado con éxito.")
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
