from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
import cv2
import io


app = FastAPI()

# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.frombuffer(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # type(cv2_img) => numpy.ndarray
    success, im = cv2.imencode('.png', cv2_img)

    return Response(content=im.tobytes(), media_type="image/png")
