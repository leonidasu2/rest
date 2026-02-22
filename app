import os
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import replicate
import requests
import uuid

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# IMPORTANTE: Pon tu API Token aquí (o mejor, en una variable de entorno)
# os.environ["REPLICATE_API_TOKEN"] = "TU_TOKEN_AQUI" 
# Si usas Render.com, pon esto en las variables de entorno del dashboard.

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No se seleccionó archivo')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='Nombre de archivo vacío')

        if file and allowed_file(file.filename):
            # 1. Guardar archivo original
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_id = str(uuid.uuid4())
            original_filename = f"original_{unique_id}.{ext}"
            restored_filename = f"restored_{unique_id}.png"
            
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], restored_filename)
            
            file.save(input_path)

            try:
                # 2. Enviar a la IA (CodeFormer)
                # Este modelo es especializado en restaurar rostros antiguos
                output = replicate.run(
                    "sczhou/codeformer:...version_hash...", # Usaremos el nombre corto abajo
                    input={
                        "image": open(input_path, "rb"),
                        "codeformer_fidelity": 0.5, # 0.0 a 1.0. Menor es más suavizado, mayor mantiene más detalle original pero menos arreglo.
                        "background_enhance": True, # Mejora el fondo también
                        "face_upsample": True, # Aumenta resolución de la cara
                        "upscale": 2 # Escala de aumento (2x)
                    }
                )
                
                # Nota: Replicate devuelve una URL de la imagen procesada.
                # Debemos descargarla y guardarla en tu servidor.
                
                img_data = requests.get(output).content
                with open(output_path, 'wb') as handler:
                    handler.write(img_data)

                return render_template('index.html', 
                                       original_img=original_filename, 
                                       restored_img=restored_filename,
                                       success=True)

            except Exception as e:
                print(f"Error IA: {e}")
                # Si la IA falla, podemos aplicar el filtro básico como respaldo
                return render_template('index.html', error=f"Error procesando con IA: {e}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)