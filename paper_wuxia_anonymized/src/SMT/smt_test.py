import argparse
import pickle
import time
from pathlib import Path
import jieba
from tqdm import tqdm
from nltk.translate import AlignedSent, ibm1, ibm2, ibm3
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUTA_DATASET = PROJECT_ROOT / "processed_data" / "wuxia_selected_100000"

PLANTILLA_MODELO = str(PROJECT_ROOT / "models" / "modelo_wuxia_{}.pkl")
PLANTILLA_SALIDA = str(PROJECT_ROOT / "src" / "SMT" / "traducciones_test_{}.txt")


def cargar_datos_seguros():
    """Loads the dataset and maneja if has splits (train/test) o is flat"""
    dataset = load_from_disk(RUTA_DATASET)
    if hasattr(dataset, 'keys'):
        nombre_particion = list(dataset.keys())[0]
        return dataset[nombre_particion], dataset
    return dataset, dataset

def entrenar(tamanio_entrenamiento, tipo_modelo):
    print(f"\n--- MODE TRAINING: {tipo_modelo.upper()} ---")
    dataset_plano, _ = cargar_datos_seguros()
    
    total_disponible = len(dataset_plano)
    if tamanio_entrenamiento == 0 or tamanio_entrenamiento > total_disponible:
        tamanio_entrenamiento = total_disponible
        
    print(f"Preparando {tamanio_entrenamiento} sentences for train...")
    dataset_train = dataset_plano.select(range(tamanio_entrenamiento))
    
    corpus_entrenamiento = []
    
    for item in tqdm(dataset_train, desc="Aligning texts"):
        try:
            ingles_crudo = item['translation']['en']
            chino_crudo = item['translation']['zh']
        except (KeyError, TypeError):
            ingles_crudo = item['en']
            chino_crudo = item['zh']

        ingles = ingles_crudo.lower().split()
        chino = list(jieba.cut(chino_crudo.replace(" ", "")))
        corpus_entrenamiento.append(AlignedSent(ingles, chino))

    print(f"\nTraining {tipo_modelo.upper()} with {len(corpus_entrenamiento)} sentences...")
    inicio = time.time()
    

    if tipo_modelo == "ibm1":
        modelo_smt = ibm1.IBMModel1(corpus_entrenamiento, 10)
    elif tipo_modelo == "ibm2":
        modelo_smt = ibm2.IBMModel2(corpus_entrenamiento, 10)
    elif tipo_modelo == "ibm3":
        modelo_smt = ibm3.IBMModel3(corpus_entrenamiento, 2)
    else:
        raise ValueError("Model not soportado.")
    
    fin = time.time()
    mins, segs = divmod(int(fin - inicio), 60)
    print(f"Training completed in {mins}m {segs}s")

    # Extract only the dictionary of translation
    tabla_limpia = {}
    for palabra_e, dict_probabilidades in modelo_smt.translation_table.items():
        tabla_limpia[palabra_e] = dict(dict_probabilidades)

    ruta_modelo = PLANTILLA_MODELO.format(tipo_modelo)
    print(f"Saving table of translation cleans in: {ruta_modelo}")
    with open(ruta_modelo, 'wb') as f:
        pickle.dump(tabla_limpia, f)
    print("Model saved successfully\n")


def inferir(tamanio_inferencia, tipo_modelo):
    print(f"\n--- MODE INFERENCE: {tipo_modelo.upper()} ---")
    ruta_modelo = PLANTILLA_MODELO.format(tipo_modelo)
    ruta_salida = PLANTILLA_SALIDA.format(tipo_modelo)
    
    print(f"Loading table statistical from {ruta_modelo}...")
    try:
        with open(ruta_modelo, 'rb') as f:
            tabla_traduccion = pickle.load(f) 
    except FileNotFoundError:
        print(f"ERROR: Was not found the model {tipo_modelo.upper()}.")
        return

    dataset_plano, dataset_completo = cargar_datos_seguros()
    
    if hasattr(dataset_completo, 'keys') and 'test' in dataset_completo.keys():
        dataset_test = dataset_completo['test']
        print("Using split 'test' oficial.")
    else:
        print("Using the final of the dataset main as test.")
        dataset_test = dataset_plano.select(range(len(dataset_plano) - 1000, len(dataset_plano)))

    total_test = len(dataset_test)
    if tamanio_inferencia == 0 or tamanio_inferencia > total_test:
        tamanio_inferencia = total_test
        
    dataset_test = dataset_test.select(range(tamanio_inferencia))
    print(f"The following will be translated: {tamanio_inferencia} sentences...")

    def traducir(frase_china, top_n=3):
        palabras_chinas = list(jieba.cut(frase_china.replace(" ", "")))
        traduccion = []
        detalles = [] 
        
        for palabra_c in palabras_chinas:
            candidatos = []
            
            # Iterate the table searching all the words in English that translate this word Chinese
            for palabra_e, probabilidades_origen in tabla_traduccion.items():
                prob = probabilidades_origen.get(palabra_c, 0.0)
                if prob > 0.0:
                    candidatos.append((palabra_e, prob))
            
            # Sort the candidates from highest to lowest probability
            candidatos_ordenados = sorted(candidatos, key=lambda x: x[1], reverse=True)
            
            if candidatos_ordenados and candidatos_ordenados[0][1] > 0.01:
                # The ganador sigue siendo the first
                mejor_palabra_e, max_prob = candidatos_ordenados[0]
                traduccion.append(mejor_palabra_e)
                
                # Build the Top N 
                top_candidatos = candidatos_ordenados[:top_n]
                strings_candidatos = [f"{pal_e}({pr * 100:.2f}%)" for pal_e, pr in top_candidatos]
                
                detalles.append(f"{palabra_c}[{'|'.join(strings_candidatos)}]")
            else:
                traduccion.append(f"[{palabra_c}]")
                detalles.append(f"[{palabra_c}](UNK)")
                
        return " ".join(traduccion), "  ".join(detalles)

    print(f"Generating output file in: {ruta_salida}")
    with open(ruta_salida, 'w', encoding='utf-8') as f_out:
        f_out.write("CHINESE_SOURCE ; ENGLISH_REFERENCE ; SMT_OUTPUT ; TOKEN_PROBABILITIES\n")
        
        for item in tqdm(dataset_test, desc="Traduciendo"):
            try:
                chino_crudo = item['translation']['zh']
                ingles_esperado = item['translation']['en']
            except (KeyError, TypeError):
                chino_crudo = item['zh']
                ingles_esperado = item['en']
                
            trad_generada, desglose_prob = traducir(chino_crudo, top_n=10) 
            
            # Cleaning
            chino_limpio = chino_crudo.replace('\n', ' ').replace(';', ',')
            ingles_limpio = ingles_esperado.replace('\n', ' ').replace(';', ',')
            trad_limpia = trad_generada.replace('\n', ' ').replace(';', ',')
            desglose_limpio = desglose_prob.replace('\n', ' ').replace(';', ',')
            
            # Format the line with the 4 columns
            linea_salida = f"{chino_limpio} ; {ingles_limpio} ; {trad_limpia} ; {desglose_limpio}"
            f_out.write(linea_salida + "\n")
            
    print(f"\nDone\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline SMT multiparadigma for TFG")
    
    parser.add_argument("mode", choices=["train", "infer"], help="Mode of execution: 'train' or 'infer'")
    
    parser.add_argument("--model", choices=["ibm1", "ibm2", "ibm3"], default="ibm2", 
                        help="Model statistical to use (default: ibm2)")
    
    parser.add_argument("--train_size", type=int, default=0, help="Number of training sentences (0 = all)")
    parser.add_argument("--infer_size", type=int, default=0, help="Number of sentences for translate in test (0 = all)")

    args = parser.parse_args()

    if args.modo == "train":
        entrenar(args.train_size, args.model)
    elif args.modo == "infer":
        inferir(args.infer_size, args.model)
        
        
        
# python test.py train --model ibm1 --train_size 5000
# python test.py infer --model ibm1 --infer_size 200
