import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUTA_MODELO = PROJECT_ROOT / "models" / "modelo_wuxia_ibm2.pkl"

print("Loading the core statistical of the model...")
with open(RUTA_MODELO, 'rb') as f:
    # dictionary: table[English][Chinese] = probability
    tabla_traduccion = pickle.load(f)

def consultar_palabra(palabra_china, top_n=5):
    """Finds all the translations possible for to word Chinese and the sorts by probability"""
    resultados = []
    
    
    for palabra_e, probabilidades_origen in tabla_traduccion.items():
        # If the word Chinese exists in the probabilities of this word inglesa
        if palabra_china in probabilidades_origen:
            prob = probabilidades_origen[palabra_china]
            # Filtramos the ruido statistical 
            if prob > 0.001: 
                resultados.append((palabra_e, prob))
                
    # Sort from highest to lowest probability
    resultados.sort(key=lambda x: x[1], reverse=True)
    
    # Return only the 'top_n' 
    return resultados[:top_n]

palabras_a_consultar = ["魔头", "杀", "剑", "宗门"] # E.g.: Demon, Kill, Sword, Sect

print("\nSTATISTICAL PROBABILITY ANALYSIS")
for palabra in palabras_a_consultar:
    opciones = consultar_palabra(palabra)
    
    print(f"\nOriginal word: 【 {palabra} 】")
    if not opciones:
        print("  -> (The model not has learned this word or the probability is almost 0)")
    else:
        for i, (traduccion, probabilidad) in enumerate(opciones, 1):
            porcentaje = probabilidad * 100
            print(f"  {i}. {traduccion:<15} -> {porcentaje:>6.2f} %")
