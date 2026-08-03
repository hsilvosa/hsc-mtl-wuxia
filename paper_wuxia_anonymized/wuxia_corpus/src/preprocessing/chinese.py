import os
import re
import argparse


def procesar_condor(nombre_archivo: str, out_dir: str):
    
    with open(nombre_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()

    patron = re.compile(r'^(第[\d一二三四五六七八九十百千万零]+回)')
    patron_en = re.compile(r'^Chapter\s+\d+', re.IGNORECASE)

    carpeta_salida = out_dir
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    contenido = []
    contador = 0  # Initialize in 0; therefore the first chapter is numbered as 1
    anterior_titulo = None

    for linea in lineas:
        linea_stripped = linea.strip()
        match = patron.match(linea_stripped)

        if patron_en.match(linea_stripped):
            continue

        if match:
            titulo_actual = match.group(1)

            if titulo_actual != anterior_titulo:
                # Save the chapter previous if already is accumulated content
                if contenido:
                    nombre_archivo_salida = os.path.join(carpeta_salida, f"{contador}ch.txt")
                    with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
                        f_out.write("".join(contenido))
                contador += 1
                contenido = [linea]
                anterior_titulo = titulo_actual
            else:
                continue  # line duplicada, ignoramos
        else:
            if contenido:
                contenido.append(linea)

    # Save the last chapter
    if contenido:
        nombre_archivo_salida = os.path.join(carpeta_salida, f"{contador}ch.txt")
        with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
            f_out.write("".join(contenido))

    print(f"{nombre_archivo} -> {contador} chapters extracted")


def procesar_guzhenren(nombre_archivo: str, out_dir: str):
    
    with open(nombre_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()

    patron = re.compile(r'^第[\d一二三四五六七八九十百千万零]+[章节](?:[:：\s])')

    carpeta_salida = out_dir
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    contenido = []
    contador = 0  # Initialize in 0; therefore the first chapter is numbered as 1
    anterior_titulo = None

    for linea in lineas:
        linea_stripped = linea.strip()
        match = patron.match(linea_stripped)

        if match:
            titulo_actual = match.group(0)  # preserves exactly it that performed the regex

            if titulo_actual != anterior_titulo:
                # Save the chapter previous if already is accumulated content
                if contenido:
                    nombre_archivo_salida = os.path.join(carpeta_salida, f"{contador}ch.txt")
                    with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
                        f_out.write("".join(contenido))
                contador += 1
                contenido = [linea]
                anterior_titulo = titulo_actual
            else:
                continue  # line duplicada, ignoramos
        else:
            if contenido:
                contenido.append(linea)

    # Save the last chapter
    if contenido:
        nombre_archivo_salida = os.path.join(carpeta_salida, f"{contador}ch.txt")
        with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
            f_out.write("".join(contenido))

    print(f"{nombre_archivo} -> {contador} chapters extracted")


def procesar_awe(nombre_archivo: str, out_dir: str):
    
    with open(nombre_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()

    # Detects lines as: 第1314章 你的选择
    patron = re.compile(r'(第[\d一二三四五六七八九十百千万零]+[章节])')

    carpeta_salida = out_dir
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    contenido = []
    contador = 0
    anterior_titulo = None

    for linea in lineas:
        linea_stripped = linea.strip()
        match = patron.match(linea_stripped)

        if match:
            titulo_actual = match.group(1)  # Only captures "第xxx章"

            if titulo_actual != anterior_titulo:
                if contenido:
                    nombre_archivo_salida = os.path.join(carpeta_salida, f"{contador}ch.txt")
                    with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
                        f_out.write("".join(contenido))
                contador += 1
                contenido = [titulo_actual + "\n"]  # Only saves "第xxx章"
                anterior_titulo = titulo_actual
            else:
                continue  # line duplicada
        else:
            if contenido:
                contenido.append(linea)

    # Save the last chapter
    if contenido:
        nombre_archivo_salida = os.path.join(carpeta_salida, f"{contador}ch.txt")
        with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
            f_out.write("".join(contenido))

    print(f"{nombre_archivo} -> {contador} chapters extracted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process novelas in Chinese and split by chapters")
    parser.add_argument("--novel", choices=["condor", "gu", "awe"], required=True,
                        help="Name of the novel to process")
    parser.add_argument("--input", required=True,
                        help="Name of the file inside of data/<novel>/raw/")
    parser.add_argument("--outname", default="segmented/chapter",
                        help="Name of the subdirectory of output inside of data/<novel>/")
    args = parser.parse_args()

    # For AWE: python scripts\preprocessing\chinese.pychinese.py --novel awe --input awe_ch.txt --outname segmented/chapter
    if args.novela == "awe":
        base_dir = os.path.join("data", args.novela, "raw")
        input_file = os.path.join(base_dir, args.input)
        out_dir = os.path.join("data", args.novela, args.outname)
        procesar_awe(input_file, out_dir)

    if args.novela == "condor":
        base_dir = os.path.join("data", args.novela, "raw")
        input_file = os.path.join(base_dir, args.input)
        out_dir = os.path.join("data", args.novela, args.outname)
        procesar_condor(input_file, out_dir)
        
    elif args.novela == "gu":
        base_dir = os.path.join("data", args.novela, "raw")
        input_file = os.path.join(base_dir, args.input)
        out_dir = os.path.join("data", args.novela, args.outname)
        procesar_guzhenren(input_file, out_dir)

