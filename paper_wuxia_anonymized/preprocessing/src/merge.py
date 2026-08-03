def unir_txts(archivos_entrada, archivo_salida):
    with open(archivo_salida, 'w', encoding='utf-8') as salida:
        for archivo in archivos_entrada:
            with open(archivo, 'r', encoding='utf-8') as entrada:
                contenido = entrada.read()
                salida.write(contenido)
                salida.write('\n')  # Adds a line in blank between files optionally

archivos = ['GU\\final_gu.txt', 'CONDOR\\final_condor_1.txt', 'CONDOR\\final_condor_1.txt', 'CONDOR\\final_condor_1.txt']#, 'AWE\\final_awe.txt']
salida = 'dataset_2.txt'
unir_txts(archivos, salida)
