import os
import re
import argparse

def corregir_espacios(palabra: str) -> str:
    return re.sub(r'(\b\w)\s+(\w\b)', r'\1\2', palabra)


def procesar_condor(nombre_archivo: str, out_dir: str):
    patron = re.compile(r'^Chapter[\s]*(\d+)[\s.:：\-–]*', re.IGNORECASE)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    contenido, cap_anterior, contador = [], None, 0

    with open(nombre_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()

    for i, linea in enumerate(lineas):
        linea_strip = linea.strip()
        match = patron.match(linea_strip)

        if match:
            cap_actual = int(match.group(1))
            if cap_actual != cap_anterior:
                if contenido:
                    cap_nombre = cap_anterior - 1 if cap_anterior >= 1215 else cap_anterior
                    nombre_archivo_salida = os.path.join(out_dir, f"{cap_nombre}en.txt")
                    with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
                        f_out.write("".join(contenido))
                contenido = [linea]
                cap_anterior = cap_actual
                contador += 1
        else:
            if contenido:
                contenido.append(corregir_espacios(linea))

    if contenido and cap_anterior is not None:
        nombre_archivo_salida = os.path.join(out_dir, f"{cap_anterior-1}en.txt")
        with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
            f_out.write("".join(contenido))

    print(f"{nombre_archivo} -> {contador} chapters in {out_dir}")


def procesar_guzhenren(nombre_archivo: str, out_dir: str):
    patron = re.compile(r'^Chapter[\s]*(\d+)[\s.:：\-–]*', re.IGNORECASE)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    contenido, cap_anterior, contador = [], None, 0

    with open(nombre_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()

    for i, linea in enumerate(lineas):
        linea_strip = linea.strip()
        match = patron.match(linea_strip)

        if match:
            cap_actual = int(match.group(1))
            if cap_actual != cap_anterior:
                if contenido:
                    cap_nombre = cap_anterior - 1 if cap_anterior >= 1215 else cap_anterior
                    nombre_archivo_salida = os.path.join(out_dir, f"{cap_nombre}en.txt")
                    with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
                        f_out.write("".join(contenido))
                contenido = [linea]
                cap_anterior = cap_actual
                contador += 1
        else:
            if contenido:
                contenido.append(corregir_espacios(linea))

    if contenido and cap_anterior is not None:
        nombre_archivo_salida = os.path.join(out_dir, f"{cap_anterior-1}en.txt")
        with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
            f_out.write("".join(contenido))

    print(f"{nombre_archivo} -> {contador} chapters in {out_dir}")


def procesar_awe(nombre_archivo: str, out_dir: str):
    patron = re.compile(r'^\d{4}\s+(chapter-\d+)', re.IGNORECASE)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    contenido, cap_anterior, contador = [], None, 0

    with open(nombre_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()

    for i, linea in enumerate(lineas):
        linea_strip = linea.strip()
        match = patron.match(linea_strip)

        if match:
            chapter_text = match.group(1)
            chapter_number = int(re.search(r'\d+', chapter_text).group())
            if chapter_number != cap_anterior:
                if contenido:
                    nombre_archivo_salida = os.path.join(out_dir, f"{cap_anterior}en.txt")
                    with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
                        f_out.write("".join(contenido))
                contenido = [chapter_text + "\n"]
                cap_anterior = chapter_number
                contador += 1
        else:
            if cap_anterior is not None:
                contenido.append(corregir_espacios(linea))

    if contenido and cap_anterior is not None:
        nombre_archivo_salida = os.path.join(out_dir, f"{cap_anterior}en.txt")
        with open(nombre_archivo_salida, 'w', encoding='utf-8') as f_out:
            f_out.write("".join(contenido))

    print(f"{nombre_archivo} -> {contador} chapters in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process novelas in English and split by chapters")
    parser.add_argument("--novel", choices=["condor", "gu", "awe"], required=True,
                        help="Name of the novel")
    parser.add_argument("--input", required=True,
                        help="Name of the file inside of data/<novel>/raw/")
    parser.add_argument("--outname", default="segmented/chapter",
                        help="Name of the subdirectory of output inside of data/<novel>/")
    args = parser.parse_args()

    # For AWE: python english.py --novel awe --input awe_en.txt --outname segmented/chapter
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
