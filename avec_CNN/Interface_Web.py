import gradio as gr
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# Désactiver les messages inutiles de TensorFlow dans le terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("⏳ Chargement du modèle en cours...")

# 1. Chargement du modèle et des classes
try:
    model = load_model('modele_reconnaissance_vocale.h5')
    classes = np.load('classes.npy')
    print("✅ Modèle et classes chargés avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")

# 2. Dictionnaire des photos
# Les clés doivent correspondre exactement aux noms des dossiers d'origine
photos_etudiants = {
    "Dounia_Zaitoune": "photos/Dounia_Zaitoune.jpeg",
    "HALIMA": "photos/HALIMA.jpeg",
    "Imane_Chalati": "photos/Imane_Chalati.jpeg",
    "Kawtar": "photos/Kawtar.jpeg",
    "Khadija": "photos/khadija.jpeg", 
    "Oumaima_Khalil": "photos/Oumaima_Khalil.jpeg",
    # Ajoute les autres étudiants ici au fur et à mesure
}

# Image par défaut si la personne n'a pas encore de photo
IMAGE_SECOURS = "photos/inconnu.jpg"

# 3. Fonction de prédiction
def predire_voix(audio_path):
    if audio_path is None:
        return "⚠️ Veuillez fournir un enregistrement audio.", IMAGE_SECOURS
        
    try:
        # --- PRÉTRAITEMENT DE L'AUDIO (Même recette que Preprocessing.py) ---
        SAMPLE_RATE = 22050
        DURATION = 3
        SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION

        # Charger l'audio
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Enlever les silences
        intervals = librosa.effects.split(signal, top_db=20)
        signal_sans_silence = []
        for start, end in intervals:
            signal_sans_silence.extend(signal[start:end])
        signal = np.array(signal_sans_silence)

        # Ajuster la taille pile à 3 secondes (Sinon le CNN va planter)
        if len(signal) > SAMPLES_PER_SEGMENT:
            signal = signal[:SAMPLES_PER_SEGMENT] # On coupe ce qui dépasse
        else:
            padding = SAMPLES_PER_SEGMENT - len(signal)
            signal = np.pad(signal, (0, padding), 'constant') # On comble avec du silence si trop court

        # Normaliser le volume
        signal = librosa.util.normalize(signal)

        # Transformer en MFCC (L'image mathématique)
        mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=13)
        
        # Ajouter les dimensions pour le CNN -> (1, 13, 130, 1)
        mfcc_final = mfcc[np.newaxis, ..., np.newaxis]

        # --- PRÉDICTION ---
        predictions = model.predict(mfcc_final)
        index_predit = np.argmax(predictions) # L'index du score le plus haut
        confiance = np.max(predictions) * 100 # Le pourcentage de certitude
        nom_predit = classes[index_predit]    # Le nom de l'étudiant

        # --- PRÉSENTATION DES RÉSULTATS ---
        # Si la confiance est trop basse, on peut considérer que la personne est inconnue
        if confiance < 50.0:
            texte_resultat = f"### 🧐 Hum...\nJe ne suis pas très sûr. Le score de **{nom_predit}** n'est que de {confiance:.1f}%."
            image_path = IMAGE_SECOURS
        else:
            texte_resultat = f"### 🎉 Résultat : **{nom_predit}**\n**Fiabilité :** {confiance:.1f}%"
            # C'est ici que l'erreur d'indentation et de parenthèse a été corrigée !
            image_path = photos_etudiants.get(nom_predit, IMAGE_SECOURS)
        
        return texte_resultat, image_path

    except Exception as e:
        return f"❌ Oups, une erreur s'est produite lors de l'analyse : {str(e)}", IMAGE_SECOURS

# 4. DESIGN DE L'INTERFACE WEB (Gradio Blocks)
theme_perso = gr.themes.Soft(
    primary_hue="indigo", 
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(title="Reconnaissance Vocale ESI") as demo:
    # En-tête
    gr.Markdown('<h1 style="text-align: center; color: #4F46E5;">🎙️ Système de Reconnaissance Vocale ESI</h1>')

# 5. Lancement du serveur Web
if __name__ == "__main__":
    print("\n🚀 Démarrage de l'interface web...")
    demo.launch(share=False, theme=theme_perso)