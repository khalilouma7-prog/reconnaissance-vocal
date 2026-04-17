import os
import librosa
import numpy as np

# Configuration
DATASET_PATH = "Data" 
SAMPLE_RATE = 22050
DURATION = 3 # secondes
SAMPLES_PER_SEGMENT = DURATION * SAMPLE_RATE

def prepare_data(dataset_path):
    X = []
    y = []
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for i, student_name in enumerate(classes):
        student_folder = os.path.join(dataset_path, student_name)
        print(f"Extraction pour : {student_name}...")
        
        for file in os.listdir(student_folder):
            # Accepte les deux formats natifs
            if file.endswith('.wav') or file.endswith('.ogg'):
                path = os.path.join(student_folder, file)
                try:
                    # Chargement (librosa gère le .ogg automatiquement)
                    signal, sr = librosa.load(path, sr=SAMPLE_RATE)
                    
                    # NOUVEAU : Suppression des silences (bruit de fond < 20dB)
                    signal, _ = librosa.effects.trim(signal, top_db=20)
                    
                    # NOUVEAU : Normalisation du volume
                    if len(signal) > 0: # Sécurité contre les fichiers vides après le trim
                        signal = librosa.util.normalize(signal)
                    
                    # Découpage en segments stricts de 3 secondes
                    for s in range(0, len(signal) - SAMPLES_PER_SEGMENT, SAMPLES_PER_SEGMENT):
                        segment = signal[s:s + SAMPLES_PER_SEGMENT]
                        
                        # Extraction MFCC
                        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                        X.append(mfcc)
                        y.append(i)
                        
                except Exception as e:
                    print(f"Erreur sur {file}: {e}")

    return np.array(X), np.array(y), classes

# Lancement de l'extraction
X, y, class_names = prepare_data(DATASET_PATH)

# Sauvegarde pour l'entraînement
np.save('X_data.npy', X)
np.save('y_data.npy', y)
np.save('classes.npy', class_names)

print(f"Prétraitement terminé ! {len(X)} segments générés pour {len(class_names)} classes.")