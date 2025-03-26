# Requisitos de uso:
Para instalar los requirements hay que utilizar pip desde un entorno de conda (.venv nos dá problemas con pandas).

## Comandos:
1. conda install pip
2. pip -r install requirements.txt

# Llamada a la plantilla:
1. Obtener una descripción sobre los argumentos de llamada:
    python3 plantilla.py --help 
2. Entrenar un modelo con la configuración por defecto de plantilla:
    python3 plantilla.py -m "train"
3. Probar el modelo:
    python3 plantilla.py -m "test"


Para cambiar la configuración solo hay que modificar config.json.
