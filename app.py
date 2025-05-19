import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import librosa
import numpy as np
# import soundfile as sf # librosa le gère pour de nombreux formats
from sklearn.preprocessing import StandardScaler
import joblib
import traceback
import sklearn
import joblib
print(f"Version Scikit-learn pour FLASK: {sklearn.__version__}")
print(f"Version Joblib pour FLASK: {joblib.__version__}")

# --- Configuration de l'Application ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'flac', 'webm'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Paramètres des Modèles et des Features ---
# Mettez à jour ces chemins pour correspondre à l'emplacement de vos fichiers de modèle sauvegardés
SVM_MODEL_PATH = 'models/emotion_svm_model.pkl'
RF_MODEL_PATH = 'models/emotion_rf_model.pkl' # Chemin pour Random Forest
LSTM_MODEL_PATH = 'models/best_lstm_model.keras' # Ou final_lstm_emotion_model.keras

# Scaler pour les features agrégées (SVM, RF)
SCALER_AGGREGATED_PATH = 'models/scaler_aggregated.pkl'
# Si vous aviez un scaler pour les features séquentielles (LSTM), ajoutez son chemin ici.
# Pour l'instant, on suppose que LSTM utilise BatchNormalization interne ou pas de scaling global.

AUDIO_DURATION = 3
AUDIO_OFFSET = 0.5 # Si appliqué PENDANT l'extraction de features
TARGET_SAMPLE_RATE = 22050

# IMPORTANT: Doit correspondre EXACTEMENT à l'ordre des classes de votre LabelEncoder
# Vérifiez la sortie de `le.classes_` dans votre notebook final.
# D'après votre dernier notebook : ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

MAX_SEQ_LENGTH = 130 # Utilisé pour les features séquentielles LSTM

# --- Chargement des Modèles et du Scaler ---
models = {}
scaler_aggregated = None

print("[INFO] Chargement des modèles...")
try:
    print(f"[INFO] Chargement du modèle SVM depuis {SVM_MODEL_PATH}...")
    models['svm'] = joblib.load(SVM_MODEL_PATH)
    print("[INFO] Modèle SVM chargé.")
except Exception as e:
    print(f"[ERREUR] Impossible de charger le modèle SVM: {e}")
    models['svm'] = None

try:
    print(f"[INFO] Chargement du modèle Random Forest depuis {RF_MODEL_PATH}...")
    models['random_forest'] = joblib.load(RF_MODEL_PATH)
    print("[INFO] Modèle Random Forest chargé.")
except Exception as e:
    print(f"[ERREUR] Impossible de charger le modèle Random Forest: {e}")
    models['random_forest'] = None

try:
    print(f"[INFO] Chargement du modèle LSTM depuis {LSTM_MODEL_PATH}...")
    models['lstm'] = tf.keras.models.load_model(LSTM_MODEL_PATH)
    print("[INFO] Modèle LSTM chargé.")
except Exception as e:
    print(f"[ERREUR] Impossible de charger le modèle LSTM: {e}")
    models['lstm'] = None

try:
    print(f"[INFO] Chargement du scaler agrégé depuis {SCALER_AGGREGATED_PATH}...")
    scaler_aggregated = joblib.load(SCALER_AGGREGATED_PATH)
    print("[INFO] Scaler agrégé chargé.")
except Exception as e:
    print(f"[ERREUR] Impossible de charger le scaler agrégé: {e}.")
    scaler_aggregated = None

if not any(models.values()):
    print("[CRITICAL] Aucun modèle n'a pu être chargé. L'application ne fonctionnera pas correctement.")
elif scaler_aggregated is None and (models.get('svm') or models.get('random_forest')):
    print("[CRITICAL] Le scaler agrégé n'a pas pu être chargé. Les prédictions SVM/RF seront incorrectes.")


# --- Fonctions d'Extraction de Features (adaptées du notebook) ---
def preprocess_audio_for_features(file_path, duration=AUDIO_DURATION, offset=AUDIO_OFFSET, target_sr=TARGET_SAMPLE_RATE):
    """Charge et prétraite l'audio pour l'extraction de features."""
    try:
        if not os.path.exists(file_path):
            print(f"[ERREUR] Fichier non trouvé à '{file_path}'")
            return None, None
        
        # Charger l'audio, resampler si nécessaire, convertir en mono
        y, sr = librosa.load(file_path, sr=target_sr, mono=True, duration=duration, offset=offset)
        
        # S'assurer que l'audio a la bonne durée (padding si plus court)
        # Cette logique est importante car les features LSTM attendent une longueur fixe de trames
        # et les features agrégées bénéficient d'une durée d'input consistante.
        expected_length = int(duration * target_sr)
        if len(y) < expected_length:
            y = np.pad(y, (0, expected_length - len(y)), 'constant')
        elif len(y) > expected_length: # Devrait être géré par duration dans librosa.load
            y = y[:expected_length]
            
        return y, sr
    except Exception as e:
        print(f"Erreur dans preprocess_audio_for_features pour {file_path}: {e}")
        traceback.print_exc()
        return None, None

def extract_aggregated_features_app(y, sr):
    """Extrait les features agrégées (pour SVM, RF)."""
    try:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        return np.concatenate((mfccs, chroma, mel, contrast, tonnetz))
    except Exception as e:
        print(f"Erreur extraction features agrégées: {e}")
        return np.zeros(173) # 20+12+128+7+6 = 173

def extract_sequential_features_app(y, sr, max_len=MAX_SEQ_LENGTH):
    """Extrait les features séquentielles (pour LSTM)."""
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T
        
        min_frames = min(mfccs.shape[0], chroma.shape[0], mel.shape[0], contrast.shape[0], tonnetz.shape[0])
        mfccs = mfccs[:min_frames, :]; chroma = chroma[:min_frames, :]
        mel = mel[:min_frames, :]; contrast = contrast[:min_frames, :]
        tonnetz = tonnetz[:min_frames, :]
        
        features_sequence = np.concatenate((mfccs, chroma, mel, contrast, tonnetz), axis=1)
        
        if features_sequence.shape[0] > max_len:
            features_sequence = features_sequence[:max_len, :]
        elif features_sequence.shape[0] < max_len:
            pad_width = max_len - features_sequence.shape[0]
            features_sequence = np.pad(features_sequence, pad_width=((0, pad_width), (0,0)), mode='constant', constant_values=0)
        return features_sequence
    except Exception as e:
        print(f"Erreur extraction features séquentielles: {e}")
        num_features_per_frame = 173
        return np.zeros((max_len, num_features_per_frame))

# --- Fonctions Utilitaires ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Routes de l'Application ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global scaler_aggregated

    if 'audio_file' not in request.files:
        return jsonify({'error': 'Aucun fichier audio trouvé.'}), 400

    file = request.files['audio_file']
    selected_model_type = request.form.get('model_type')

    if not selected_model_type or selected_model_type not in models or models[selected_model_type] is None:
        return jsonify({'error': f'Modèle "{selected_model_type}" non spécifié, invalide ou non chargé.'}), 400
    
    current_model = models[selected_model_type]
    model_display_name = selected_model_type.upper() # Pour l'affichage

    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné.'}), 400

    if file and allowed_file(file.filename):
        filepath = None # Initialiser filepath
        try:
            filename = "uploaded_audio." + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            y_processed, sr_processed = preprocess_audio_for_features(filepath)
            if y_processed is None:
                if filepath and os.path.exists(filepath): os.remove(filepath)
                return jsonify({'error': 'Erreur lors du prétraitement audio.'}), 500

            predicted_emotion = "Inconnu"
            confidence_str = ""
            probabilities_dict = {}

            if selected_model_type in ['svm', 'random_forest']:
                if scaler_aggregated is None:
                    if filepath and os.path.exists(filepath): os.remove(filepath)
                    return jsonify({'error': 'Scaler agrégé non chargé.'}), 500
                
                features_agg = extract_aggregated_features_app(y_processed, sr_processed)
                if np.all(features_agg == 0): print("[WARNING] Features agrégées nulles.")
                
                features_scaled_agg = scaler_aggregated.transform([features_agg])[0]
                
                if selected_model_type == 'svm':
                    model_display_name = "SVM"
                else: # random_forest
                    model_display_name = "Random Forest"

                if hasattr(current_model, 'predict_proba'):
                    pred_index_array = current_model.predict([features_scaled_agg])
                    pred_index = int(pred_index_array[0]) if isinstance(pred_index_array, np.ndarray) else int(pred_index_array)
                    
                    probs_array = current_model.predict_proba([features_scaled_agg])[0]
                    confidence = float(probs_array[pred_index] * 100)
                    confidence_str = f"{confidence:.2f}%"
                    # Assurez-vous que EMOTION_LABELS correspond à model.classes_ pour SVM/RF
                    # Si current_model.classes_ existe et est différent, il faut mapper.
                    # Pour la simplicité, on assume que l'ordre de EMOTION_LABELS est le bon.
                    probabilities_dict = {EMOTION_LABELS[i]: f"{float(probs_array[i]*100):.2f}%" for i in range(len(probs_array))}
                else: # Si pas de predict_proba (ne devrait pas arriver avec SVC(probability=True))
                    pred_index_array = current_model.predict([features_scaled_agg])
                    pred_index = int(pred_index_array[0]) if isinstance(pred_index_array, np.ndarray) else int(pred_index_array)
                    confidence_str = "N/A (pas de probas)"
                    probabilities_dict = {label: "N/A" for label in EMOTION_LABELS}
                    probabilities_dict[EMOTION_LABELS[pred_index]] = "100.00% (classe prédite)"
                
                predicted_emotion = EMOTION_LABELS[pred_index]

            elif selected_model_type == 'lstm':
                model_display_name = "LSTM"
                features_seq = extract_sequential_features_app(y_processed, sr_processed)
                if np.all(features_seq == 0): print("[WARNING] Features séquentielles nulles.")
                
                # LSTM attend une entrée (batch_size, timesteps, features)
                features_seq_batch = np.expand_dims(features_seq, axis=0)
                
                predictions_probs_all = current_model.predict(features_seq_batch)[0]
                predicted_index = np.argmax(predictions_probs_all)
                confidence = float(predictions_probs_all[predicted_index] * 100)
                confidence_str = f"{confidence:.2f}%"
                probabilities_dict = {label: f"{float(prob*100):.2f}%" for label, prob in zip(EMOTION_LABELS, predictions_probs_all)}
                predicted_emotion = EMOTION_LABELS[predicted_index]

            if filepath and os.path.exists(filepath): os.remove(filepath)

            return jsonify({
                'emotion': predicted_emotion.upper(),
                'confidence': confidence_str,
                'probabilities': probabilities_dict,
                'model_used_display': model_display_name
            })

        except Exception as e:
            print(f"Erreur PRED Endpoint: {e}")
            traceback.print_exc()
            if filepath and os.path.exists(filepath): os.remove(filepath)
            return jsonify({'error': f'Erreur serveur : {str(e)}'}), 500
    else:
        return jsonify({'error': 'Type de fichier non autorisé.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)