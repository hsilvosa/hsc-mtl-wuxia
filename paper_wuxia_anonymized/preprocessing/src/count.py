
import os

def contar_palabras(archivo):
    """Count the words in to text file."""
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            texto = f.read()
            palabras = texto.split()
            return len(palabras)
    except Exception as e:
        print(f"Error processing '{archivo}': {e}")
        return None

if __name__ == "__main__":
    # Corpus paralelo: .../data/inputs/corpus
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inputs", "corpus")

    print("Word count by file:\n" + "-"*40)

    archivos_txt = [f for f in os.listdir(carpeta) if f.lower().endswith(".txt")]

    if not archivos_txt:
        print("No files were found txt")
    else:
        for nombre in archivos_txt:
            ruta = os.path.join(carpeta, nombre)
            cantidad = contar_palabras(ruta)
            if cantidad is not None:
                print(f"{nombre}: {cantidad} words")

