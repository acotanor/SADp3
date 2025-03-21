import pandas as pd
import sys
import nltk
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

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
        
        # Categorical features
        categorical_feature = data.select_dtypes(include='object').columns.tolist()
        
        # Text features
        text_feature = [col for col in categorical_feature if col != 'v1']
        
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

if __name__ == '__main__':
    # Configurar argumentos
    parser = argparse.ArgumentParser(description="Procesamiento de datos")
    parser.add_argument('--file', type=str, required=True, help="Ruta del archivo CSV")
    parser.add_argument('--text_process', type=str, choices=['tf-idf', 'bow'], required=True, help="Técnica de procesamiento de texto a utilizar")
    parser.add_argument('--output', type=str, required=True, help="Ruta del archivo CSV de salida")
    parser.add_argument('--columna', type=str, required=True, help="Columna a tratar.")
    args = parser.parse_args()

    # Carga de datos
    data = load_data(args.file)
    
    # Seleccionar características
    numerical_feature, text_feature, categorical_feature = select_features(data)
    
    # Mostrar las características
    print("Características numéricas:", numerical_feature)
    print("Características de texto:", text_feature)
    print("Características categóricas:", categorical_feature)
    
    # Procesar valores faltantes
    data = process_missing_values(data, numerical_feature, categorical_feature)
    
    # Simplificar texto
    data = simplify_text(data, text_feature)
    
    # Mostrar la columna de texto simplificada
    print("Texto simplificado en la columna " + args.columna + ":")
    print(data[args.columna].head(10))  # Mostrar las primeras 10 filas de la columna v2 simplificada
    
    # Procesar texto
    process_text(text_feature, args.text_process)
    
    # Guardar datos procesados en un nuevo archivo CSV
    data.to_csv(args.output, index=False)
    print(f"Datos procesados guardados en {args.output}")
