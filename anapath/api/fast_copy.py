from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from anapath.Logic.registry import load_model

from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import io


app = FastAPI()

# Allow all requests (optional, good for development purposes)
app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],  # Allows all origins
     allow_credentials=True,
     allow_methods=["*"],  # Allows all methods
     allow_headers=["*"],  # Allows all headers
)

app.state.modeldiag = load_model("production","diag")
app.state.modeltx = load_model("production","diag")

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.frombuffer(contents, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    model_diag = app.state.modeldiag
    model_tx = app.state.modeltx

    img = cv2.resize(img_cv2,(512, 256))
    #img = np.array(img)
    image_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_array = np.expand_dims(image_array, axis=0)
    datagen = ImageDataGenerator(rescale=1./255)
    img_generator = datagen.flow(
        image_array,
        batch_size=1,
        shuffle=False )


    proba_diag = model_diag.predict(img_generator)[0][0]
    result_ = "Pas de cancer"
    if proba_diag >= 0.85:
        proba_tx = model_tx.predict(img_generator)[0][0]
        if proba_tx >= 0.85:
            taux = 'High'
        else:
            taux = 'low'
        result_ = f"Suspicion de cancer ({proba_diag*100}%) avec un taux de cellularité tumoral {proba_tx*100}%"

    success, im = cv2.imencode('.png', img_cv2)
    headers = {
        "X-Prediction-Result": result_  # Ajouter le résultat dans un en-tête personnalisé
    }

    return Response(
        content=im.tobytes(),
        media_type="image/png",
        headers=headers)
