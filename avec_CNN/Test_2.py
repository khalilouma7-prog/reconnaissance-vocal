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
    SAMPLES_PER_SEGMENT = 3 * fs # 66150 échantillons (exactement 3 secondes)
    
    print(f"\n>>> Analyse du fichier : {chemin_fichier}")
    
    if not os.path.exists(chemin_fichier):
        print("Erreur : Le fichier n'existe pas. Vérifiez le chemin.")
        return

    try:
        # --- 1. Chargement et Nettoyage ---
        signal, sr = librosa.load(chemin_fichier, sr=fs)
        signal, _ = librosa.effects.trim(signal, top_db=20)
        
        # --- 2. LA MODIFICATION CRUCIALE ---
        # On vérifie qu'on a bien au moins 3 secondes de voix utile
        if len(signal) < SAMPLES_PER_SEGMENT:
            print("Erreur : L'audio est trop court après avoir enlevé les silences. Il faut au moins 3 secondes de parole.")
            return
            
        # On bloque la taille à EXACTEMENT 66150 échantillons (comme à l'entraînement)
        signal = signal[:SAMPLES_PER_SEGMENT]
        
        # --- 3. Normalisation ---
        signal = librosa.util.normalize(signal)
        
        # --- 4. Extraction MFCC ---
        # Comme le signal fait exactement 3s, la matrice fera naturellement (13, 130)
        mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13)
        
        # Préparation du format pour le CNN (Batch, 13, 130, 1)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        
        # --- 5. Prédiction ---
        prediction = model.predict(mfcc, verbose=0)
        index = np.argmax(prediction)
        probabilite = np.max(prediction)
        
        # Affichage du résultat
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