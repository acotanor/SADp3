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
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
    """
    Carga los datos desde un archivo CSV y elimina columnas innecesarias.

    Args:
        file (str): Ruta del archivo CSV.
        encoding (str): Codificación del archivo (por defecto 'utf-8').

    Returns:
        pd.DataFrame: DataFrame con los datos cargados y limpiados.
    """
    try:
        # Cargar el archivo CSV
        data = pd.read_csv(file, encoding=encoding)
        
        # Eliminar columnas innecesarias
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
        return data
    except UnicodeDecodeError:
        print(f"Error al decodificar el archivo con la codificación {encoding}. Intentando con 'latin1'.")
        data = pd.read_csv(file, encoding='latin1')
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
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
        
        # Categorical features
        categorical_feature = data.select_dtypes(include='object').columns.tolist()
        
        # Text features: Detectar columnas con texto largo (por ejemplo, más de 30 caracteres en promedio)
        text_feature = [col for col in categorical_feature if data[col].str.len().mean() > 30]
        
        # Excluir columnas de texto de las categóricas
        categorical_feature = [col for col in categorical_feature if col not in text_feature]

        print("Características numéricas identificadas:", numerical_feature)
        print("Características categóricas identificadas:", categorical_feature)
        print("Características de texto identificadas:", text_feature)
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
        numerical_feature (array-like): El array que contiene las características numéricas.

    Returns:
        numerical_feature: El dato de entrada reescalado.

    Raises:
        Exception: Si hay un error al reescalar los datos.
    """
    # Verificar si el array está vacío
    if numerical_feature.size == 0:  # Cambiado de .empty a .size
        print("No se encontraron características numéricas para reescalar.")
        return numerical_feature

    scaler = MinMaxScaler(feature_range=(0, 1))
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

    Args:
        text_feature (list): Lista de nombres de las columnas de texto.
        text_process (str): Técnica de procesamiento de texto a utilizar ("tf-idf" o "bow").
    """
    global data
    try:
        if len(text_feature) > 0:
            for feature in text_feature:
                if text_process == "tf-idf":
                    tfidf_vectorizer = TfidfVectorizer()
                    tfidf_matrix = tfidf_vectorizer.fit_transform(data[feature])
                    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
                    data = pd.concat([data.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
                    data.drop(columns=[feature], inplace=True)
                    print(f"Texto de la columna '{feature}' procesado con éxito usando TF-IDF")
                elif text_process == "bow":
                    bow_vectorizer = CountVectorizer()
                    bow_matrix = bow_vectorizer.fit_transform(data[feature])
                    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
                    data = pd.concat([data.reset_index(drop=True), bow_df.reset_index(drop=True)], axis=1)
                    data.drop(columns=[feature], inplace=True)
                    print(f"Texto de la columna '{feature}' procesado con éxito usando Bag of Words")
                else:
                    print(f"No se está procesando el texto de la columna '{feature}'")
        else:
            print("No se han encontrado columnas de texto a procesar")
    except Exception as e:
        print("Error al procesar el texto")
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
    
    # Convertir variables categóricas a numéricas
    data = cat2num(data, categorical_feature)
    
    # Simplificar texto
    if text_feature:
        data = simplify_text(data, text_feature)
    
    # Procesar texto
    if text_feature:
        process_text(text_feature, text_process)
    
    # Reescalar características numéricas
    if numerical_feature:
        print("Reescalando características numéricas...")
        data[numerical_feature] = reescaler(data[numerical_feature])
    else:
        print("No se encontraron características numéricas para reescalar.")
    
    print("Tipos de datos en el DataFrame:")
    print(data.dtypes)
    print("Columnas del DataFrame después de categorizar:", data.columns)

#    __  __           _      _           
#   |  \/  | ___   __| | ___| | ___  ___ 
#   | |\/| |/ _ \ / _` |/ _ \ |/ _ \/ __|
#   | |  | | (_) | (_| |  __/ | (_) \__ \
#   |_|  |_|\___/ \__,_|\___|_|\___/|___/

from sklearn.preprocessing import LabelEncoder

def divide_data():
    """
    Divide los datos en conjuntos de entrenamiento y desarrollo.
    Codifica las etiquetas (y) como valores numéricos.
    """
    global data
    X = data.drop(columns=[args.column])  # Eliminar la columna objetivo
    y = data[args.column]  # Columna objetivo

    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Dividir los datos en entrenamiento y desarrollo
    x_train, x_dev, y_train, y_dev = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state  # Usar random_state desde el JSON
    )

    # Convertir X a arrays de NumPy para evitar problemas con nombres de columnas
    x_train = x_train.values
    x_dev = x_dev.values

    return x_train, x_dev, y_train, y_dev

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

def mostrar_resultados(modelo, X, y_true):
    """
    Muestra los resultados del modelo utilizando las etiquetas reales.
    """
    try:
        # Realizar predicciones
        y_pred = modelo.predict(X)

        # Calcular métricas
        print("> Informe de clasificación:\n", classification_report(y_true, y_pred))
        print("> Matriz de confusión:\n", confusion_matrix(y_true, y_pred))
        print("> F1-score micro:", f1_score(y_true, y_pred, average='micro'))
        print("> F1-score macro:", f1_score(y_true, y_pred, average='macro'))
    except Exception as e:
        print("Error al evaluar el modelo:")
        print(e)

def calculate_fscore(y_true, y_pred):
    """
    Calcula el F1-score de un clasificador. 
    """
    return f1_score(y_true, y_pred, average='micro'), f1_score(y_true, y_pred, average='macro')

def knn():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparámetros para encontrar los parámetros óptimos.
    """
    # Dividimos los datos en entrenamiento y desarrollo
    x_train, x_dev, y_train, y_dev = divide_data()

    # Reescalamos los datos
    x_train = reescaler(x_train)
    x_dev = reescaler(x_dev)
    
    # Configuramos el barrido de hiperparámetros
    param_grid = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    gs = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=args.cpu)

    # Entrenamos el modelo
    gs.fit(x_train, y_train)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.
    Realiza un barrido de hiperparámetros para encontrar los mejores parámetros.
    """
    # Dividimos los datos en entrenamiento y desarrollo
    x_train, x_dev, y_train, y_dev = divide_data()

    # Configuramos el barrido de hiperparámetros
    param_grid = {
        "criterion": [args.criterion],  # Usar el criterio desde el JSON
        "max_depth": [None, 5, 10, 15, 20],  # Profundidad máxima del árbol
        "min_samples_split": [2, 5, 10],  # Mínimo número de muestras para dividir un nodo
        "min_samples_leaf": [1, 2, 4]  # Mínimo número de muestras en una hoja
    }

    # Configuramos GridSearchCV
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=args.random_state),  # Usar random_state desde el JSON
        param_grid,
        n_jobs=args.cpu,
        cv=5
    )

    # Entrenamos el modelo
    print("Entrenando el modelo Decision Tree...")
    gs.fit(x_train, y_train)

    # Guardamos el modelo utilizando pickle
    save_model(gs)

def random_forest():
    """
    Función para implementar el algoritmo Random Forest.
    Realiza un barrido de hiperparámetros utilizando GridSearchCV para encontrar los mejores parámetros.
    """
    # Dividimos los datos en entrenamiento y desarrollo
    x_train, x_dev, y_train, y_dev = divide_data()

    # Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        "n_estimators": [10, 50, 100],  # Número de árboles en el bosque
        "max_depth": [None, 10, 20],  # Profundidad máxima del árbol
        "min_samples_split": [2, 5],  # Mínimo número de muestras para dividir un nodo
        "min_samples_leaf": [1, 2],  # Mínimo número de muestras en una hoja
        "bootstrap": [True, False]  # Si se utiliza muestreo con reemplazo
    }

    # Configurar GridSearchCV
    gs = GridSearchCV(
        estimator=RandomForestClassifier(random_state=args.random_state),  # Usar random_state desde el JSON
        param_grid=param_grid,
        cv=5,  # Validación cruzada con 5 particiones
        n_jobs=args.cpu,  # Número de CPUs a utilizar
        verbose=2  # Nivel de detalle en la salida
    )

    print("Entrenando el modelo Random Forest con GridSearchCV...")
    gs.fit(x_train, y_train)

    # Guardar el modelo utilizando pickle
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

def predict(modelo, y_true=None):
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.
    Si se proporcionan etiquetas reales (y_true), evalúa el modelo.

    Parámetros:
        modelo: El modelo entrenado.
        y_true: Las etiquetas reales (opcional).

    Retorna:
        Ninguno
    """
    global data
    # Eliminar la columna objetivo del conjunto de características
    X = data.drop(columns=[args.column])

    # Convertir X a un array de NumPy para evitar inconsistencias
    X = X.values

    # Verificar que X y y_true tienen el mismo tamaño
    if y_true is not None and X.shape[0] != len(y_true):
        print("Error: El número de filas en X no coincide con el número de etiquetas reales (y_true).")
        return

    # Realizar predicciones
    prediction = modelo.predict(X)

    # Añadir las predicciones al DataFrame
    data['Predicción'] = prediction

    # Guardar el DataFrame con las predicciones
    data.to_csv('output/data-prediction.csv', index=False)
    print("Predicciones guardadas en 'output/data-prediction.csv'")

    # Evaluar el modelo si se proporcionan etiquetas reales
    if y_true is not None:
        print("\n> Evaluando el modelo...")
        mostrar_resultados(modelo, X, y_true)

#    __  __       _       
#   |  \/  | __ _(_)_ __  
#   | |\/| |/ _` | | '_ \ 
#   | |  | | (_| | | | | |
#   |_|  |_|\__,_|_|_| |_|
                    
if __name__ == '__main__':
    # Cargar configuración desde config.json
    config_file = "config.json"
    config = leerJson(config_file)

    if not config:
        print("Error: No se pudo cargar la configuración desde config.json.")
        sys.exit(1)

    # Asignar valores desde el archivo JSON
    jsonArgs = argparse.Namespace(
        data=config.get("data"),
        text_process=config.get("text_process"),
        sampling=config.get("sampling"),
        output=config.get("output"),
        column=config.get("column"),
        modelo=config.get("modelo"),
        cpu=config.get("cpu", -1),
        test=config.get("test", 0.25),
        criterion=config.get("criterion", "gini"),  # Valor por defecto: "gini"
        random_state=config.get("random_state", 42)  # Valor por defecto: 42
    )
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m","--mode", type=str, choices=['train','test'], help="Modo de ejecución (train/test)", required=True)

    args = parser.parse_args(namespace=jsonArgs)

    np.random.seed(args.random_state)  # Utilizamos la semilla desde el JSON para reproducibilidad.

    # Crear la carpeta output si no existe
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

    # Generar configuración si no existe
    generarJson()

    # Cargar datos
    print(f"\n- Cargando datos desde {args.data}...")
    data = load_data(args.data)

    # Descargar recursos necesarios de NLTK
    print("\n- Descargando diccionarios...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Preprocesar datos
    preprocesar_datos(args.text_process)

    # Guardar datos procesados en un nuevo archivo CSV
    data.to_csv(args.output, index=False)
    print(f"Datos procesados guardados en {args.output}")

    if args.mode == 'train':
        # Entrenar el modelo seleccionado
        print(f"Entrenando el modelo {args.modelo}...")
        if args.modelo == "knn":
            try:
                knn()
                print("Modelo KNN entrenado con éxito.")
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.modelo == "dt":
            try:
                decision_tree()
                print("Modelo Decision Tree entrenado con éxito.")
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.modelo == "rf":
            try:
                random_forest()
                print("Modelo Random Forest entrenado con éxito.")
                sys.exit(0)
            except Exception as e:
                print(e)
    elif args.mode == 'test':
        print(f"\n- Cargando el modelo {args.modelo}...")
        modelo = load_model()
        try:
            # Obtener las etiquetas reales
            y_true = data[args.column].values

            # Realizar predicciones y evaluar el modelo
            predict(modelo, y_true)
            print(f"Test del modelo {args.modelo} realizado con éxito.")
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
