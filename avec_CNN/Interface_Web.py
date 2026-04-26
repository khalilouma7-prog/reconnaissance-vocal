import gradio as gr
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─────────────────────────────────────────────
# 1. CHARGEMENT DU MODÈLE ET DES CLASSES
# ─────────────────────────────────────────────
print("⏳ Chargement du modèle en cours...")
try:
    model   = load_model('modele_reconnaissance_vocale.h5')
    classes = np.load('classes.npy')
    print("✅ Modèle et classes chargés avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model, classes = None, None

# ─────────────────────────────────────────────
# 2. DICTIONNAIRE DES PHOTOS
# ─────────────────────────────────────────────
photos_etudiants = {
    "Dounia_Zaitoune" : "photos/Dounia_Zaitoune.jpeg",
    "HALIMA"          : "photos/HALIMA.jpeg",
    "Imane_Chalati"   : "photos/Imane_Chalati.jpeg",
    "Kawtar"          : "photos/Kawtar.jpeg",
    "Khadija"         : "photos/khadija.jpeg",
    "Oumaima_Khalil"  : "photos/Oumaima_Khalil.jpeg",
    "Hiba_EL_Idrisy"  : "photos/Hiba_EL_Idrisy.jpeg",
    "Khadija_EL_Boudhiri" : "photos/Khadija_EL_Boudhiri.jpeg",
    "Salma_Nrhira" : "photos/Salma_Nrhira.jpeg"
    }
IMAGE_SECOURS = "photos/inconnu.jpg"

# ─────────────────────────────────────────────
# 3. FONCTION DE RECONNAISSANCE DU LOCUTEUR
# ─────────────────────────────────────────────
def identifier_locuteur(audio_path):
    if audio_path is None:
        return "⚠️ Veuillez fournir un enregistrement audio.", IMAGE_SECOURS

    if model is None or classes is None:
        return "❌ Le modèle n'est pas chargé.", IMAGE_SECOURS

    try:
        SAMPLE_RATE         = 22050
        DURATION            = 3
        SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION

        signal, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Suppression des silences
        intervals = librosa.effects.split(signal, top_db=20)
        signal_propre = []
        for start, end in intervals:
            signal_propre.extend(signal[start:end])
        signal = np.array(signal_propre) if signal_propre else signal

        # Ajustement à 3 secondes exactes
        if len(signal) > SAMPLES_PER_SEGMENT:
            signal = signal[:SAMPLES_PER_SEGMENT]
        else:
            signal = np.pad(signal, (0, SAMPLES_PER_SEGMENT - len(signal)), 'constant')

        # Normalisation + MFCC
        signal     = librosa.util.normalize(signal)
        mfcc       = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=13)
        mfcc_final = mfcc[np.newaxis, ..., np.newaxis]

        # Prédiction
        predictions  = model.predict(mfcc_final)
        index_predit = np.argmax(predictions)
        confiance    = float(np.max(predictions)) * 100
        nom_predit   = classes[index_predit]

        if confiance < 50.0:
            texte      = (f"### 🧐 Locuteur inconnu\n"
                          f"Ressemble à **{nom_predit}** mais confiance trop faible : **{confiance:.1f}%**")
            image_path = IMAGE_SECOURS
        else:
            texte      = (f"### ✅ Locuteur identifié : **{nom_predit}**\n"
                          f"**Confiance :** {confiance:.1f}%")
            image_path = photos_etudiants.get(nom_predit, IMAGE_SECOURS)

        return texte, image_path

    except Exception as e:
        return f"❌ Erreur : {str(e)}", IMAGE_SECOURS

# ─────────────────────────────────────────────
# 4. INTERFACE
# ─────────────────────────────────────────────
theme_perso = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(title="Reconnaissance du Locuteur — ESI") as demo:

    gr.Markdown("""
    <h1 style="text-align:center; color:#4F46E5;">🎙️ Reconnaissance du Locuteur — ESI</h1>
    <p style="text-align:center; color:#64748B;">
        Parlez dans le micro ou uploadez un fichier audio — le système identifie qui parle.
    </p>
    """)

    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Enregistrement")
            audio_input  = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Parlez ici"
            )
            btn_identifier = gr.Button("🔍 Identifier le locuteur", variant="primary", size="lg")
            btn_effacer    = gr.Button("🗑️ Effacer", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 👤 Résultat")
            texte_output = gr.Markdown("_En attente d'un enregistrement…_")
            image_output = gr.Image(label="Photo du locuteur", height=280)

    btn_identifier.click(
        fn=identifier_locuteur,
        inputs=audio_input,
        outputs=[texte_output, image_output]
    )

    btn_effacer.click(
        fn=lambda: (None, "_En attente d'un enregistrement…_", None),
        inputs=None,
        outputs=[audio_input, texte_output, image_output]
    )

    gr.Markdown("<p style='text-align:center; color:#94A3B8; font-size:0.85rem;'>Projet Reconnaissance du Locuteur — ESI</p>")

# ─────────────────────────────────────────────
# 5. LANCEMENT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Démarrage...")
    demo.launch(share=False, theme=theme_perso)