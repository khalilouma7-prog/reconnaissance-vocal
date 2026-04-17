import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- CONFIGURATION ---
DATASET_PATH = "Dataset" # Ton dossier principal
IMG_SHAPE = (13, 130)    # Taille fixe pour le CNN (13 coeffs, ~3 sec d'audio)
SAMPLES_PER_SEGMENT = 3 * 22050 # 3 secondes à 22.05kHz

def prepare_data(dataset_path):
    X = []
    y = []
    classes = os.listdir(dataset_path)
    
    for i, student_name in enumerate(classes):
        student_folder = os.path.join(dataset_path, student_name)
        if not os.path.isdir(student_folder): continue
        
        print(f"Traitement de : {student_name}...")
        
        for file in os.listdir(student_folder):
            if file.endswith('.wav'):
                file_path = os.path.join(student_folder, file)
                
                # Charger l'audio
                signal, sr = librosa.load(file_path, sr=22050)
                
                # Découper en segments de 3 secondes pour avoir + de données
                for s in range(0, len(signal) - SAMPLES_PER_SEGMENT, SAMPLES_PER_SEGMENT):
                    segment = signal[s:s + SAMPLES_PER_SEGMENT]
                    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                    
                    # On s'assure que la taille est toujours identique
                    if mfcc.shape[1] == IMG_SHAPE[1]:
                        X.append(mfcc)
                        y.append(i)
                        
    return np.array(X), np.array(y), classes

# Exécution du chargement
X, y, class_names = prepare_data(DATASET_PATH)

# Normalisation et mise en forme pour le CNN
X = X[..., np.newaxis] # Ajout de la dimension "canal" (comme une image NB)
y = to_categorical(y, num_classes=len(class_names))

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_model(input_shape, num_classes):
    model = Sequential([
        # Bloc 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Bloc 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Classification
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # Softmax pour les 9 étudiants
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Création et entraînement
model = build_model(X_train.shape[1:], len(class_names))
model.summary()

history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=30, 
                    batch_size=8)

import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np

def reconnaitre_par_micro(model, encoder, duration=4, fs=48000, output_file="test_micro.wav"):
    print("Parle maintenant...")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    print("Enregistrement terminé.")

    write(output_file, fs, recording)

    audio, sr = librosa.load(output_file, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_mean = mfcc_mean.reshape(1, -1)

    prediction = model.predict(mfcc_mean)
    predicted_class = np.argmax(prediction)
    predicted_name = encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction)

    print("Personne reconnue :", predicted_name)
    print("Confiance :", confidence)

    return predicted_name, confidence