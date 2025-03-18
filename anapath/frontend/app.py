import streamlit as st
from PIL import Image
import requests
import os
import cv2
import numpy as np

# Set page tab display
st.set_page_config(
   page_title="Simple Image Uploader",
   page_icon= '🖼️',
   layout="wide",
   initial_sidebar_state="expanded",
)

# Example local Docker container URL
# url = 'http://api:8000'
# Example localhost development URL
# url = 'http://localhost:8000'
#load_dotenv()
#url = os.getenv('API_URL')
url = 'https://anapath-1068402267466.europe-west1.run.app'

# App title and description
st.header('Diagnostic Anapath 📸')
st.markdown('''
            > Outil d'aide à la décision d'images Anapath
            > Merci de rentrer une photo de tissu pour que l'outil sorte
            > un diagnostic de cancer et le taux de cellularité tumoral
            ''')

st.markdown("---")

### Create a native Streamlit file upload input
st.markdown("### Merci d'uploader votre fichier ici pour l'envoyer 👇")
img_file_buffer = st.file_uploader('Upload an image')

if img_file_buffer is not None:

  col1, col2 = st.columns(2)

  with col1:
    ### Display the image user uploaded
    st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")

  with col2:
    with st.spinner("Wait for it..."):
      ### Get bytes from the file buffer
      img_bytes = img_file_buffer.getvalue()

      # Convert bytes to numpy array
      nparr = np.frombuffer(img_bytes, np.uint8)
      img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

      # If you need RGB instead of BGR (OpenCV uses BGR by default)
      img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
      ### Make request to  API (stream=True to stream response as bytes)
      res = requests.post(url + "/upload_image", files={'img': img_bytes})

      if res.status_code == 200:
        ### Display the image returned by the API
        st.image(res.content, caption="Image returned from API ☝️")
      else:
        st.markdown("**Oops**, something went wrong 😓 Please try again.")
        print(res.status_code, res.content)
