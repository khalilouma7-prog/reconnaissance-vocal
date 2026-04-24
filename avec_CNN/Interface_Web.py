import gradio as gr
import numpy as np
import librosa
import os

# Désactiver les messages TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model

print("Chargement du modèle...")
model = load_model('modele_reconnaissance_vocale.h5')
class_names = np.load('classes.npy')
print("Modèle chargé ! Lancement de l'interface...")

# Fonction qui sera appelée quand on clique sur "Envoyer" dans l'interface
def reconnaitre_voix(audio_path):
    if audio_path is None:
        return "⚠️ Veuillez enregistrer un audio."

    fs = 22050
    SAMPLES_PER_SEGMENT = 3 * fs # 3 secondes

    try:
        # Chargement de l'audio venant de l'interface web
        signal, sr = librosa.load(audio_path, sr=fs)
        signal, _ = librosa.effects.trim(signal, top_db=20)

        # Vérification de la durée
        if len(signal) < SAMPLES_PER_SEGMENT:
            return "❌ Erreur : L'audio est trop court. Parlez pendant au moins 3 à 4 secondes."

        # Découpage, normalisation et MFCC (La méthode parfaite)
        signal = signal[:SAMPLES_PER_SEGMENT]
        signal = librosa.util.normalize(signal)
        mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]

        # Prédiction
        prediction = model.predict(mfcc, verbose=0)
        index = np.argmax(prediction)
        probabilite = np.max(prediction)

        if probabilite < 0.40:
            return "❓ RÉSULTAT : Locuteur inconnu (Confiance trop faible)."
        else:
            return f"✅ RÉSULTAT : {class_names[index]} (Confiance : {probabilite*100:.1f}%)"
            
    except Exception as e:
        return f"❌ Erreur technique : {str(e)}"

# --- CRÉATION DE L'INTERFACE VISUELLE ---
interface = gr.Interface(
    fn=reconnaitre_voix,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Enregistrez votre voix (Parlez 4 secondes)"),
    outputs=gr.Textbox(label="🧠 Prédiction de l'Intelligence Artificielle", lines=2),
    title="Système de Reconnaissance Automatique du Locuteur",
    description="Cliquez sur le micro, parlez normalement, puis cliquez sur Soumettre pour laisser le CNN (Deep Learning) analyser votre empreinte vocale.",
    theme="default"
)

# Lancement du serveur local
if __name__ == "__main__":
    interface.launch()