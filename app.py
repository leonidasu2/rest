import os
import uuid
import requests
import replicate
from flask import Flask, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return "API Restaurador IA Activa"


@app.route("/api/restaurar", methods=["POST"])
def restaurar():

    if "file" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Archivo vacío"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Formato no permitido"}), 400

    if "REPLICATE_API_TOKEN" not in os.environ:
        return jsonify({"error": "Falta REPLICATE_API_TOKEN"}), 500

    try:
        # Generar nombres únicos
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_id = str(uuid.uuid4())

        original_filename = f"original_{unique_id}.{ext}"
        restored_filename = f"restored_{unique_id}.png"

        original_path = os.path.join(UPLOAD_FOLDER, original_filename)
        restored_path = os.path.join(UPLOAD_FOLDER, restored_filename)

        # Guardar imagen original
        file.save(original_path)

        # 🔥 Llamada moderna a Replicate (versión fija estable)
        output = replicate.run(
            "sczhou/codeformer:latest",
            input={
                "image": open(original_path, "rb"),
                "codeformer_fidelity": 0.7,
                "background_enhance": True,
                "face_upsample": True,
                "upscale": 2,
            },
        )

        # El modelo devuelve una URL
        img_url = output

        response = requests.get(img_url)

        if response.status_code != 200:
            return jsonify({"error": "Error descargando imagen IA"}), 500

        # Guardar imagen restaurada
        with open(restored_path, "wb") as f:
            f.write(response.content)

        # Crear URL pública
        public_url = request.host_url + "static/uploads/" + restored_filename

        return jsonify({"restored_url": public_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()