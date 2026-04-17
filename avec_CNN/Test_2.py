import numpy as np
import librosa
import os

# Désactiver les messages d'avertissement TensorFlow dans la console (optionnel)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model

# 1. Charger le modèle et les noms des classes
print("Chargement du modèle...")
model = load_model('modele_reconnaissance_vocale.h5')
class_names = np.load('classes.npy')
print("Modèle chargé avec succès !\n")

def tester_fichier_audio(chemin_fichier):
    fs = 22050
    print(f"\n>>> Analyse du fichier : {chemin_fichier}")
    
    if not os.path.exists(chemin_fichier):
        print("Erreur : Le fichier n'existe pas. Vérifiez le chemin.")
        return

    try:
        # --- 1. Chargement du fichier (Accepte .wav et .ogg) ---
        signal, sr = librosa.load(chemin_fichier, sr=fs)
        
        # --- 2. PRÉTRAITEMENT EXACTEMENT IDENTIQUE À L'ENTRAÎNEMENT ---
        
        # Trim avec le même seuil (suppression des silences)
        signal, _ = librosa.effects.trim(signal, top_db=20)
        
        # Normalisation du volume audio
        if len(signal) > 0:
            signal = librosa.util.normalize(signal)
        else:
            print("Erreur : Fichier audio vide après la suppression du silence.")
            return
        
        # Extraction MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13)
        
        # --- 3. Forcer la largeur à 130 pour correspondre au modèle CNN ---
        expected_width = 130
        current_width = mfcc.shape[1]
        
        if current_width > expected_width:
            mfcc = mfcc[:, :expected_width] # Tronquer si trop long
        else:
            pad_width = expected_width - current_width
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant') # Padder si trop court
        
        # Préparation du format pour Keras (Batch, 13, 130, 1)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        
        # --- 4. Prédiction ---
        prediction = model.predict(mfcc, verbose=0)
        index = np.argmax(prediction)
        probabilite = np.max(prediction)
        
        # Affichage du résultat avec seuil de sécurité
        if probabilite < 0.40:
            print("RÉSULTAT : Locuteur inconnu (Confiance trop faible).")
        else:
            print(f"RÉSULTAT : {class_names[index]} ({probabilite*100:.1f}%)")
            
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")

# Boucle interactive pour tester plusieurs fichiers à la suite
while True:
    print("-" * 50)
    chemin = input("Entrez le chemin du fichier audio à tester (ou 'q' pour quitter) : ")
    
    # Enlever les guillemets si on glisse-dépose le fichier dans le terminal
    chemin = chemin.strip('"').strip("'")
    
    if chemin.lower() == 'q': 
        print("Fin du test.")
        break
        
    tester_fichier_audio(chemin)