"""Flask routes and startup."""
import io

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from model_service import ModelService
from recommendations import get_recommendation

MAX_UPLOAD_MB = 5
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs

service = ModelService()
service.load()  # load once at startup


def error(message, status):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_ready": service.ready, "detail": service.error})


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return error("No image uploaded.", 400)

    # Read into memory: nothing is saved to disk, so filenames are never used.
    data = file.read()
    try:
        probe = Image.open(io.BytesIO(data))
        if probe.format not in ALLOWED_FORMATS:
            return error("Unsupported format. Use JPG, PNG or WEBP.", 415)
        probe.verify()
        img = Image.open(io.BytesIO(data))  # reopen after verify()
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return error("The file is not a valid image.", 400)

    if not service.ready:
        return error("Model is not available. Train it first (see README).", 503)

    try:
        result = service.predict(img)
    except Exception:
        return error("Prediction failed. Please try another image.", 500)

    result["recommendation"] = get_recommendation(result["category"])
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return error(f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.", 413)


if __name__ == "__main__":
    app.run(debug=False)
