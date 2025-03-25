import streamlit as st
from PIL import Image
import requests
import os
import numpy as np
from io import BytesIO
import time
import json

# Configuration de la page
st.set_page_config(
    page_title="Analyse Histopathologique",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Personnalisation CSS pour améliorer l interface
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title-container {
        background-color: #0077b6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .subtitle {
        color: #023e8a;
        margin-bottom: 1rem;
    }
    .section-title {
        color: #023e8a;
        border-bottom: 2px solid #0077b6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .upload-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .result-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background-color: #e9f5fd;
        border-left: 5px solid #0077b6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.8rem;
    }
    .loader {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# URL de l API
url = 'https://anapath2-1068402267466.europe-west1.run.app'
#url = 'http://localhost:8000'

# En-tête et introduction
st.markdown('<div class="title-container"><h1>Diagnostic Histopathologique Assisté par IA</h1></div>', unsafe_allow_html=True)

st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
### 🔍 À propos de cet outil
Cet outil d aide à la décision analyse les images histologiques et fournit une évaluation :
- Du diagnostic potentiel de tissus cancéreux
- De l estimation du taux de cellularité tumorale
- Des caractéristiques morphologiques significatives

**Important :** Cet outil est conçu pour assister les professionnels de santé et ne remplace pas l expertise médicale.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Section de téléchargement d image
st.markdown('<h2 class="section-title">Téléchargement d\'Image</h2>', unsafe_allow_html=True)
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### Veuillez télécharger une image de tissu histologique")
st.markdown("Pour des résultats optimaux, utilisez des images de haute qualité au format JPG ou PNG.")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# Section d analyse et résultats
if uploaded_file is not None:
    st.markdown('<h2 class="section-title">Analyse et Résultats</h2>', unsafe_allow_html=True)
    st.markdown('<div class="result-section">', unsafe_allow_html=True)

    # Disposition en colonnes pour l image originale et les résultats
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="subtitle">Image Source</h3>', unsafe_allow_html=True)

        # Affichage de l image téléchargée
        image = Image.open(uploaded_file)
        st.image(image, caption="Image histologique téléchargée", use_container_width=True)

        # Informations sur l image
        st.markdown(f"**Dimensions:** {image.width} × {image.height} pixels")
        st.markdown(f"**Format:** {uploaded_file.type}")

    with col2:
        st.markdown('<h3 class="subtitle">Résultats de l\'Analyse</h3>', unsafe_allow_html=True)

        # Simulation de chargement avec une barre de progression
        with st.spinner("Analyse en cours..."):
            # Obtention des bytes de l image
            img_bytes = uploaded_file.getvalue()

            # Appel à l API
            try:
                # Afficher une barre de progression pour améliorer l expérience utilisateur
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simulation de traitement
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                # Envoi de l image à l API
                res = requests.post(url + "/upload_image", files={'img': img_bytes}, timeout=30)

                if res.status_code == 200:
                    # Affichage de l image annotée retournée par l API
                    #st.image(res.content, caption="Image Analysée avec Annotations", use_container_width=True)

                    # Ici, vous pourriez ajouter plus de données de résultat si votre API les retourne
                    # Par exemple, un JSON avec les résultats d'analyse détaillés
                    st.markdown('<h3 class="subtitle">Résultats de l\'Analyse</h3>', unsafe_allow_html=True)

                    result_data = res.headers
                    st.markdown("### **Microbiopsie d'une lésion du sein gauche (externe) :**")
                    st.markdown("")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Diagnostic:** {result_data.get('X-Prediction-Result')}")
                    if res.headers.get('p_class_d') == 'tumor':
                        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;carcinome infiltrant de type non spécifique - Échantillon tumoral inclus en paraffine pour génétique somatique")
                        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Absence de facteur confondant à type de nécrose, fibrose, ou mucine.")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Intervalle de confiance pour le diagnostic :** {result_data.get('c_diag')} %")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Pourcentage de cellules tumorales dans la zone sélectionnée {result_data.get('p_class_tx').upper()}**")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Intervalle de confiance pour le Taux de Cellularité:** {result_data.get('c_tx')} %")
                    else:

                    # Affichage des résultats détaillés
                    st.markdown("### Résultats Détaillés de l'Analyse")
                    #st.json(result_data)


                    st.success("Analyse complétée avec succès")


                    # Avertissement médical
                    st.warning("**Remarque importante :** Ces résultats sont générés automatiquement et doivent être confirmés par un anatomopathologiste.")

                    # Section de feedback
                    st.markdown("### Évaluation de la qualité du résultat")
                    st.markdown("Votre retour nous aide à améliorer la précision de notre outil. Ce résultat vous semble-t-il correct?")

                    # Utilisation de la session state pour suivre l état du feedback
                    if 'feedback_submitted'  not in st.session_state:
                        st.session_state.feedback_submitted = False
                    if 'show_negative_form' not in st.session_state:
                        st.session_state.show_negative_form = False

                    # Fonction pour gérer le feedback positif
                    def submit_positive_feedback():
                        try:
                            feedback_data = {
                                "image_id": uploaded_file.name,
                                "feedback": "positive",
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }

                            feedback_res = requests.post(url + "/feedback", json=feedback_data)
                            if feedback_res.status_code == 200:
                                st.session_state.feedback_submitted = True
                                st.session_state.feedback_success = True
                            else:
                                st.session_state.feedback_error = f"Erreur: {feedback_res.status_code} - {feedback_res.text}"
                        except Exception as e:
                            st.session_state.feedback_error = str(e)

                    # Fonction pour afficher le formulaire de feedback négatif
                    def show_negative_feedback_form():
                        st.session_state.show_negative_form = True

                    # Fonction pour envoyer le feedback négatif
                    def submit_negative_feedback():
                        try:
                            feedback_data = {
                                "image_id": uploaded_file.name,
                                "feedback": "negative",
                                "comment": st.session_state.negative_comment,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }

                            feedback_res = requests.post(url + "/feedback", json=feedback_data)
                            if feedback_res.status_code == 200:
                                st.session_state.feedback_submitted = True
                                st.session_state.feedback_success = True
                            else:
                                st.session_state.feedback_error = f"Erreur: {feedback_res.status_code} - {feedback_res.text}"
                        except Exception as e:
                            st.session_state.feedback_error = str(e)

                    # Affichage des boutons ou du résultat du feedback
                    if st.session_state.feedback_submitted:
                        if 'feedback_success' in st.session_state and st.session_state.feedback_success:
                            st.success("Merci pour votre retour! Il nous aidera à améliorer notre système.")
                        elif 'feedback_error' in st.session_state:
                            st.error(f"Erreur lors de l envoi du retour: {st.session_state.feedback_error}")
                    else:
                        col_thumbs_up, col_thumbs_down = st.columns(2)

                        with col_thumbs_up:
                            st.button("👍 Résultat correct", on_click=submit_positive_feedback)

                        with col_thumbs_down:
                            st.button("👎 Résultat incorrect", on_click=show_negative_feedback_form)

                        # Affichage du formulaire de feedback négatif si demandé
                        if st.session_state.show_negative_form:
                            st.text_area("Pourriez-vous préciser ce qui semble incorrect?", key="negative_comment", height=100)
                            st.button("Envoyer votre commentaire", on_click=submit_negative_feedback)

                else:
                    st.error(f"Erreur lors de l analyse: {res.status_code} - {res.text}")

            except Exception as e:
                st.error(f"Une erreur s'est produite: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# Section d informations supplémentaires
st.markdown('<h2 class="section-title">Informations Complémentaires</h2>', unsafe_allow_html=True)
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
### Méthodologie
Cette application utilise un modèle d intelligence artificielle entraîné sur des milliers d images histopathologiques annotées par des experts. l algorithme analyse les caractéristiques morphologiques des tissus pour identifier les patterns associés aux différents types de néoplasies.

### Confidentialité
Toutes les images téléchargées sont traitées de manière sécurisée et ne sont pas conservées au-delà de la session d analyse.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Pied de page
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
© 2024 Service d Anatomie Pathologique - Tous droits réservés
Cet outil est destiné à un usage professionnel uniquement.
""")
st.markdown('</div>', unsafe_allow_html=True)
