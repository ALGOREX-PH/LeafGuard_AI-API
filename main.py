from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

model_path = "Apple_Leaf_Disease_Classification.h5"
model = load_model(model_path)

def size_regulator(image_bytes, target_size=(100, 100)):
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("L")
        resized_image = pil_image.resize(target_size)
        img_array = np.array(resized_image).reshape(1, 100, 100, 1)
        return img_array
    except Exception as e:
        raise ValueError(f"Error in image preprocessing: {e}")

def predict(image_bytes):
    processed_image = size_regulator(image_bytes)
    prediction = np.argmax(model.predict(processed_image))
    classes = ["Healthy", "Apple Scab", "Black Rot", "Cedar Apple Rust"]
    return classes[prediction] if prediction < len(classes) else "Unknown"

@app.post("/predict")
async def classify_image(file: UploadFile):
    try:
        image_bytes = await file.read()
        result = predict(image_bytes)
        return JSONResponse(content={"prediction": result})
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": "Failed to process the image."}, status_code=500)

@app.get("/")
def root():
    return {"message": "Welcome to the Apple Leaf Disease Classifier API"}
