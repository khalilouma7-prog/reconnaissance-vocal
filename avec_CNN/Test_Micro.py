import sounddevice as sd
import numpy as np
import librosa
import os

# Désactiver les messages d'avertissement TensorFlow dans la console (optionnel)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model

# Charger le modèle et les noms
model = load_model('modele_reconnaissance_vocale.h5')
class_names = np.load('classes.npy')

def identifier_locuteur():
    fs = 22050
    duration = 3
    print("\n>>> Parlez maintenant (3 secondes)...")
    
    # 1 SEUL ENREGISTREMENT
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    
    print("Analyse de la voix...")
    signal = recording.flatten()
    
    # --- PRÉTRAITEMENT EXACTEMENT IDENTIQUE À L'ENTRAÎNEMENT ---
    
    # 1. Trim avec le même seuil
    signal, _ = librosa.effects.trim(signal, top_db=20)
    
    # 2. Normalisation du volume audio
    if len(signal) > 0:
        signal = librosa.util.normalize(signal)
    
    # 3. Extraction MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13)
    
    # --- Forcer la largeur à 130 pour correspondre au modèle ---
    expected_width = 130
    current_width = mfcc.shape[1]
    
    if current_width > expected_width:
        mfcc = mfcc[:, :expected_width] # Tronquer
    else:
        pad_width = expected_width - current_width
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant') # Padder
    
    # Préparation pour le CNN (Batch, 13, 130, 1)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    
    # Prédiction
    prediction = model.predict(mfcc, verbose=0) # verbose=0 cache la barre de progression keras
    index = np.argmax(prediction)
    probabilite = np.max(prediction)
    
    # Affichage
    if probabilite < 0.40:
        print("RÉSULTAT : Locuteur inconnu ou bruit trop important.")
    else:
        print(f"RÉSULTAT : {class_names[index]} ({probabilite*100:.1f}%)")

# Boucle de test
while True:
    identifier_locuteur()
    recommencer = input("Appuyez sur Entrée pour tester à nouveau (ou 'q' pour quitter) : ")
    if recommencer.lower() == 'q': 
        break
