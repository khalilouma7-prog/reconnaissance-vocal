import os
import librosa
import numpy as np

# --- PARAMÈTRES GLOBAUX ---
DATASET_PATH = 'Data'          # Le dossier qui contient tes amis
SAMPLE_RATE = 22050            # Qualité audio standard
DURATION = 3                   # La taille du segment pour le CNN (3 secondes)
SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION

# LE MULTIPLICATEUR : Au lieu d'avancer de 3s, on avance de 1s (Chevauchement)
# Cela va multiplier tes données par 3 automatiquement !
HOP_LENGTH = SAMPLES_PER_SEGMENT // 3 

def prepare_data(dataset_path):
    data = []
    labels = []
    classes = []

    print("🚀 Début du prétraitement avancé (Extraction vocale + Data Augmentation)")
    print("-" * 50)

    # 1. Parcourir les dossiers des étudiants
    for index, student_name in enumerate(os.listdir(dataset_path)):
        student_folder = os.path.join(dataset_path, student_name)

        # On ignore les fichiers perdus, on ne veut que les dossiers
        if not os.path.isdir(student_folder):
            continue
        
        classes.append(student_name)
        print(f"🎙️ Extraction en cours pour : {student_name}...")

        # 2. Parcourir les audios de chaque étudiant
        for file in os.listdir(student_folder):
            if file.endswith(('.wav', '.m4a')):
                path = os.path.join(student_folder, file)
                
                try:
                    # Chargement du fichier audio
                    signal, sr = librosa.load(path, sr=SAMPLE_RATE)

                    # --- SUPER-POUVOIR 1 : SUPPRESSION DE TOUS LES SILENCES ---
                    # Coupe l'audio dès que le volume descend sous 20 décibels
                    intervals = librosa.effects.split(signal, top_db=20)
                    signal_sans_silence = []
                    for start, end in intervals:
                        signal_sans_silence.extend(signal[start:end])
                    
                    # On remplace l'ancien signal par le signal "voix pure"
                    signal = np.array(signal_sans_silence)

                    # Sécurité : Si l'audio est trop court après suppression du silence
                    if len(signal) < SAMPLES_PER_SEGMENT:
                        print(f"  ⚠️ {file} est trop court après avoir enlevé les silences.")
                        continue

                    # --- SUPER-POUVOIR 2 : DÉCOUPAGE AVEC CHEVAUCHEMENT ---
                    # On découpe l'audio tous les 'HOP_LENGTH' (1 seconde)
                    for s in range(0, len(signal) - SAMPLES_PER_SEGMENT, HOP_LENGTH):
                        segment = signal[s : s + SAMPLES_PER_SEGMENT]
                        
                        # Normalisation (met toutes les voix au même niveau sonore)
                        segment = librosa.util.normalize(segment)

                        # Transformation en image mathématique (MFCC)
                        mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=13)
                        
                        data.append(mfcc)
                        labels.append(index)
                
                except Exception as e:
                    print(f"  ❌ Erreur ignorée sur le fichier {file} : {e}")

    # 3. Formatage pour le Réseau de Neurones (CNN)
    X = np.array(data)
    y = np.array(labels)
    
    # On ajoute la dimension "canal" pour le CNN Keras (comme une image en noir et blanc)
    X = X[..., np.newaxis] 

    # 4. Sauvegarde des fichiers
    np.save('data_X.npy', X)
    np.save('data_y.npy', y)
    np.save('classes.npy', np.array(classes))

    print("-" * 50)
    print(f"✅ Prétraitement terminé avec succès !")
    print(f"📊 Segments de voix pure générés : {len(data)} (Pour {len(classes)} personnes).")
    print("🧠 Tu peux maintenant lancer Training.py !")

if __name__ == "__main__":
    prepare_data(DATASET_PATH)