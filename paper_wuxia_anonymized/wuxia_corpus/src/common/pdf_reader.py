import PyPDF2
import re

def detectar_titulo_y_limpieza(texto):
    lineas = texto.split('\n')
    resultado = ''
    buffer_titulo = ''
    recolectando_titulo = False

    for linea in lineas:
        linea = linea.strip()

        # Ignorar lines empty or basura
        if not linea or 'OceanofPDF' in linea:
            continue

        # If detect start of chapter
        if re.match(r'^Chapter\s+\d+\s*:', linea, re.IGNORECASE):
            if buffer_titulo:
                resultado += '\n\n' + buffer_titulo.strip() + '\n\n'
                buffer_titulo = ''
            buffer_titulo = linea
            recolectando_titulo = True
            continue

        # If are recolectando title (lines extra as "Mount Heaven...")
        if recolectando_titulo:
            if re.match(r'^[A-Z][\w\s,\-]+$', linea) and len(linea.split()) <= 10:
                buffer_titulo += ' ' + linea
                continue
            else:
                resultado += '\n\n' + buffer_titulo.strip() + '\n\n'
                buffer_titulo = ''
                recolectando_titulo = False

        # Reconstruct sentences
        if resultado and not resultado.endswith(('\n\n', '.', '!', '?', ':', '"')):
            resultado += ' ' + linea
        else:
            resultado += linea + '\n'

    if buffer_titulo:
        resultado += '\n\n' + buffer_titulo.strip() + '\n\n'

    return re.sub(r'[ \t]+', ' ', resultado.strip())

def pdf_a_txt_con_titulos(ruta_pdf, ruta_txt):
    try:
        with open(ruta_pdf, 'rb') as archivo_pdf:
            lector = PyPDF2.PdfReader(archivo_pdf)
            texto_total = ""

            for pagina in lector.pages:
                texto = pagina.extract_text()
                if texto:
                    texto_total += texto + '\n'

        texto_limpio = detectar_titulo_y_limpieza(texto_total)

        with open(ruta_txt, 'w', encoding='utf-8') as archivo_txt:
            archivo_txt.write(texto_limpio)

        print(f"Conversion completed with detection of multiple title formats. File: {ruta_txt}")
    except Exception as e:
        print(f"Error: {e}")


pdf_a_txt_con_titulos("i_shall_seal_the_heavens_removed.pdf", "issth_en.txt")
