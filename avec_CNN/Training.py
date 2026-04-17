import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Désactiver les avertissements inutiles dans le terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("--- 1. Chargement des données ---")
X = np.load('X_data.npy')
y = np.load('y_data.npy')
classes = np.load('classes.npy')

print(f"Données chargées : {len(X)} échantillons pour {len(classes)} étudiants.")

# Reformater X pour le CNN (Ajout de la dimension du canal, ex: (N, 13, 130) -> (N, 13, 130, 1))
if len(X.shape) == 3:
    X = X[..., np.newaxis]

# Convertir les labels en probabilités (One-Hot Encoding)
y = to_categorical(y, num_classes=len(classes))

# Séparer : 80% pour apprendre, 20% pour vérifier sans tricher
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- 2. Création du modèle Anti-Surapprentissage ---")
model = Sequential([
    # Bloc 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3), # Oublie 30%

    # Bloc 2
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4), # Oublie 40%

    # Décision
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5), # Oublie 50%

    # Sortie (Un neurone par étudiant)
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- 3. Lancement de l'entraînement ---")

# Arrêter si la validation (val_accuracy) ne s'améliore plus après 12 époques
early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1)

# Sauvegarder automatiquement la meilleure version
model_checkpoint = ModelCheckpoint('modele_reconnaissance_vocale.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Lancer l'apprentissage
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, # On met 100, mais EarlyStopping l'arrêtera avant
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

print("\n✅ Entraînement terminé avec succès !")
print("Le réseau a été sauvegardé sous 'modele_reconnaissance_vocale.h5'. Tu peux lancer tes tests.")