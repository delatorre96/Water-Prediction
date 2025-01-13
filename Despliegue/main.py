import logging
from functions import flujoTrabajo

def main():
    # Configuración básica de logging para mostrar mensajes en la consola
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("""Bienvenido al programa de predicción de cantidad de agua. Este programa ha sido desarrollado por el Grupo de Economía del Agua del IIDL: Institut de Desenvolupament Local de la Universitat de València. Su principal objetivo es elaborar una predicción de la cantidad de caudal que hay en la Cuenca Hidrográfica del Jucar utilizando datos meteorológicos.""")
    print("Por favor, ingrese el número de días a predecir:")
    
    # Capturar entrada del usuario
    while True:
        try:
            retardosMax = int(input("Días a predecir: "))
            if retardosMax <= 0:
                raise ValueError("El número de días debe ser un entero positivo.")
            break
        except ValueError as e:
            print(f"Entrada inválida: {e}. Intente nuevamente.")
    
    # Llamar a la función principal del flujo de trabajo
    logging.info(f"Comenzando el flujo de trabajo para {retardosMax} días...")
    flujoTrabajo(retardosMax)
    logging.info("Predicción completada con éxito.")

if __name__ == "__main__":
    main()