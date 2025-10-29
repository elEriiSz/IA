Programas, archivos y video de presentacion

Los programas principales son los de titulo gen_dataset.py, ia_trainer2.py y simple_program2.py.
Los archivos de importancia son pupil_tracker.keras ya que contiene el modelo entrenado y training_metrics que muestra las metricas del entrenamiento
El dataset esta guardado como un directorio bajo el nombre de dataset. Este sigue el siguiente modelo:

'''
dataset/
 ├── left/
 │    ├── images/
 │    └── labels/
 └── right/
      ├── images/
      └── labels/
'''

Los frameworks y librerias usadas fueron numpy, opencv, scikit, tensorflow, matplot, mediapipe y pytesseract
Para que el dataset quede limpio, lo que se utilizo fue el script de python gen_dataset.py el cual automatizo el proceso de guardado y etiquetado de cada dato

La presentacion:
https://docs.google.com/presentation/d/1fdtN2oAKJ04yIHb7a5JYXmXetYq-EAE36F6MwSudRAc/edit?usp=sharing

El video de la presentacion:
https://youtu.be/WfEp7fL5EOg


