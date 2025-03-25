from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from pathlib import Path
from PIL import Image
import numpy as np
from peft import PeftModel
import cv2
import os
from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
import torch



app = FastAPI()

# Allow all requests (optional, good for development purposes)
app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],  # Allows all origins
     allow_credentials=True,
     allow_methods=["*"],  # Allows all methods
     allow_headers=["*"],  # Allows all headers
)

#################################################################
model_name = "owkin/phikon-v2"
model_directory = Path("anapath/Models")
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
image_size = image_processor.crop_size if "v2" in model_name else image_processor.size
label2id_diag = {0: "Normal", 1: "Tumor"}
id2label_diag = {v: k for k, v in label2id_diag.items()}
label2id_tx = {0: "Low", 1: "High"}
id2label_tx = {v: k for k, v in label2id_tx.items()}
model_diag = AutoModelForImageClassification.from_pretrained(
    model_name,
    label2id=label2id_diag,
    id2label=id2label_diag,
    ignore_mismatched_sizes=False,
)
model_tx = AutoModelForImageClassification.from_pretrained(
    model_name,
    label2id=label2id_tx,
    id2label=id2label_tx,
    ignore_mismatched_sizes=False,
)
app.state.model_diag = PeftModel.from_pretrained(
    model_diag,
    os.path.join(model_directory,"diag")
)
app.state.model_tx = PeftModel.from_pretrained(
    model_tx,
    os.path.join(model_directory,"tx")
)


#################################################################

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.frombuffer(contents, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Alternative: spécifier explicitement le device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_diag = app.state.model_diag.to(device)
    model_tx = app.state.model_tx.to(device)


    #img = cv2.resize(img_cv2,(512, 256))
    #img = np.array(img)
    image_array = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    #image_array = np.expand_dims(image_array, axis=0)
    #datagen = ImageDataGenerator(rescale=1./255)
    #img_generator = datagen.flow(
    #    image_array,
    #    batch_size=1,
    #    shuffle=False )
    #################################################################

    #image = Image.open(f"{testing_directory}/tumor/{file}")
    inputs_diag = image_processor(images=image_array, return_tensors="pt").to(model_diag.device)
    inputs_tx = image_processor(images=image_array, return_tensors="pt").to(model_tx.device)
    with torch.no_grad():
        outputs_tx = model_tx(**inputs_tx, output_hidden_states=True)
        outputs_diag = model_diag(**inputs_diag, output_hidden_states=True)

    # Récupérer l'embedding CLS (utile pour la classification)
    #embedding = outputs.hidden_states[-1][:, 0, :].detach().cpu().numpy()
    probabilities_diag = torch.softmax(outputs_diag.logits, dim=1)
    probabilities_tx = torch.softmax(outputs_tx.logits, dim=1)

    predicted_class_idx_diag = probabilities_diag.argmax(dim=1).item()
    predicted_class_idx_tx = probabilities_tx.argmax(dim=1).item()
    predicted_class_diag = label2id_diag[predicted_class_idx_diag].lower()
    predicted_class_tx = label2id_tx[predicted_class_idx_tx].lower()
    confidence_diag = probabilities_diag[0, predicted_class_idx_diag].item()
    confidence_tx = probabilities_tx[0, predicted_class_idx_tx].item()


    #################################################################


    #result_ = "Pas de cancer"
    if predicted_class_diag == "tumor":

        if predicted_class_tx == "high":
            class_ = 'High'
            diag = 'Tumor'
        else:
            taux = 'low'
        result_ = f"Suspicion de cancer avec un intervalle de confiance {round(confidence_diag*100,2)}% et avec un taux de cellularité tumoral {predicted_class_tx} avec un intervalle de confiance de {round(confidence_tx*100,2)} %"
    else:
        result_ = f"Absence de tissu tumoral avec un intervalle de confiance de {round(confidence_diag*100,2)}%"
    success, im = cv2.imencode('.png', img_cv2)
    headers = {
        "X-Prediction-Result": result_ , # Ajouter le résultat dans un en-tête personnalisé
        "p_diag" : str(probabilities_diag),
        "p_tx" : str(probabilities_tx),
        "p_class_d" : str(predicted_class_diag),
        "p_class_tx" : str(predicted_class_tx),
        "c_diag" : str(confidence_diag),
        "c_tx" : str(confidence_tx),
    }

    return Response(
        content=im.tobytes(),
        media_type="image/png",
        headers=headers)
