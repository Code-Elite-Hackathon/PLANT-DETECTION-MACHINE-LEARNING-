# PLANT-DETECTION-MACHINE-LEARNING-
from flask import Flask, request, jsonify, render_template, url_for
import os
from tensorflow.keras.models import load_model
import numpy as np
import sqlite3
from PIL import Image
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path="/static")

# ====================== CONFIGURATION ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
# Path to plants images dataset
DATASET_PATH = r"C:\Users\Hardeep Singh\Desktop\plant_dataSet"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = ( r"D:\Major Project Plant Detection WebApp\src\model\plant_detection_model.h5")
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_detection_model.h5")
# Path to SQLite Database file
DB_PATH = "D:\Major Project Plant Detection WebApp\plants.db"
# Add this to your app.py after the DB_PATH definition
print("\n=== Database Verification ===")
print(f"Database exists: {os.path.exists(DB_PATH)}")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size

# ====================== MODEL LOADING ======================
try:
    model = load_model(MODEL_PATH)
    class_names: list[str] = os.listdir ("C:\\Users\Hardeep Singh\OneDrive\Desktop\plant_dataSet")
    print(f"✅ Model loaded successfully from {"C:\\Users\Hardeep Sroject Plant Detection WebApp\src\model\plant_detection_model.h5"}")
    print(f"✅ Class names: {class_names}")
except Exception as e:
    print(f"\n\n----->Failed to load model: {str(e)}")
    raise e
    # class_names = []
    # model = None


# ====================== HELPER FUNCTIONS ======================
def preprocess_image(image):
    """Resize and normalize image for model prediction"""
    image = image.resize((224, 224))
    return np.expand_dims(np.array(image) / 255.0, axis=0)


def get_plant_info(plant_name):
    """Fetch plant details from database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM plants WHERE LOWER(name) = LOWER(?)", (plant_name,)
            )
            plant = cursor.fetchone()

            # Trying to retrieve info using scintific name if info is not found using plant folder name in above
            if not plant:
                cursor.execute(
                    "SELECT * FROM plants WHERE LOWER(name) = LOWER(?)", (plant_name,)
                )

                plant = cursor.fetchone()

            if plant:
                plant_data = dict(plant)
                # Convert comma-separated strings to lists
                plant_data["uses"] = (
                    plant_data["uses"].split(", ") if plant_data["uses"] else []
                )
                plant_data["medical_uses"] = (
                    plant_data["medical_uses"].split(", ")
                    if plant_data["medical_uses"]
                    else []
                )
                plant_data["common_locations"] = {
                    "punjab": plant_data.pop("location_punjab", "Not specified"),
                    "global": plant_data.pop("location_global", "Not specified"),
                }
                return plant_data
            return {"error": "Plant not found", "name": plant_name}

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return {"error": "Database error", "name": plant_name}


def clear_uploads_folder():
    """Delete all files in uploads folder"""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


# ====================== ROUTES ======================
@app.route("/")
def home():
    """Render main page"""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload and prediction"""

    # Clear ALL previous uploads first
    clear_uploads_folder()

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded", "status": "error"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected", "status": "error"}), 400

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plant_{timestamp}.jpg"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Save file
        file.save(save_path)
        print(f"Image saved to: {save_path}")

        # Process image
        img = Image.open(save_path).convert("RGB")
        if model:
            prediction = model.predict(preprocess_image(img))
            plant_name = class_names[np.argmax(prediction)]
            plant_info = get_plant_info(plant_name)
        else:
            raise Exception("Model not loaded")

        return jsonify(
            {
                "plant_name": plant_name,
                "plant_info": plant_info,
                "temp_image_url": url_for("static", filename=f"uploads/{filename}"),
                "status": "success",
            }
        )

    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise e
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/upload_camera", methods=["POST"])
def upload_camera():
    if "camera_image" not in request.files:
        return {"success": False, "message": "No file uploaded"}

    file = request.files["camera_image"]
    if file.filename == "":
        return {"success": False, "message": "No selected file"}

    if file and allowed_file(file.filename):
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_{timestamp}_{secure_filename(file.filename)}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # Return relative path for web access
        return {
            "success": True,
            "imagePath": url_for("static", filename=f"uploads/{filename}"),
        }

    return {"success": False, "message": "Invalid file type"}


@app.route("/debug")
def debug():
    """Debug endpoint to verify configuration"""
    return jsonify(
        {
            "upload_folder": app.config["UPLOAD_FOLDER"],
            "folder_exists": os.path.exists(app.config["UPLOAD_FOLDER"]),
            "static_folder": app.static_folder,
            "model_loaded": model is not None,
            "class_names_count": len(class_names),
        }

    )


# ====================== MAIN ======================
if __name__ == "__main__":
    print("\n=== Configuration ===")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Database path: {DB_PATH}")
    print("====================\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
