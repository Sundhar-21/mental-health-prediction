import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import contextlib
import json
import sys
import gc
import wave

# Extreme Memory Optimization for Render Free Tier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
gender_model = None
emotion_model = None
face_cascade = None
eye_cascade = None

# Model Classes
GENDER_CLASSES = ["male", "female"]
EMOTION_CLASSES = ["happy", "sad", "angry", "neutral", "fear"]

def load_models():
    """Load AI models for gender and emotion detection"""
    global gender_model, emotion_model, face_cascade, eye_cascade
    try:
        # Load cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Load Keras models if files exist
        if os.path.exists("keras_model.h5"):
            gender_model = load_model("keras_model.h5", compile=False)
            print("✅ Gender Model loaded.")
        else:
            print("⚠️ Gender Model not found.")

        if os.path.exists("keras_modelemo.h5"):
            emotion_model = load_model("keras_modelemo.h5", compile=False)
            print("✅ Emotion Model loaded.")
        else:
            print("⚠️ Emotion Model not found.")
            
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")

# Avoid loading models on startup to prevent timeout on Render
# with app.app_context():
#     load_models()

def get_face_cascade():
    global face_cascade
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def get_eye_cascade():
    global eye_cascade
    if eye_cascade is None:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return eye_cascade

def analyze_face_image(image_path):
    """Analyze face image for gender and emotion with minimal memory footprint"""
    results = {}
    try:
        from tf_keras import backend as K
        from tf_keras.models import load_model

        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not read image'}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade_local = get_face_cascade()
        faces = face_cascade_local.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
             return {'warning': 'No face detected'}
        
        # Process the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_roi = img[y:y+h, x:x+w]
        
        # Preprocess for models
        face_resized = cv2.resize(face_roi, (224, 224))
        input_image = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        input_image = (input_image / 127.5) - 1
        
        # 1. Gender Prediction
        if os.path.exists("keras_model.h5"):
            print("⏳ Loading Gender Model...")
            sys.stdout.flush()
            model = load_model("keras_model.h5", compile=False)
            gender_pred = model.predict(input_image, verbose=0)
            results['gender'] = GENDER_CLASSES[np.argmax(gender_pred)]
            print("✅ Gender Predicted. Unloading...")
            del model
            K.clear_session()
            gc.collect()
            sys.stdout.flush()

        # 2. Emotion Prediction
        if os.path.exists("keras_modelemo.h5"):
            print("⏳ Loading Emotion Model...")
            sys.stdout.flush()
            model = load_model("keras_modelemo.h5", compile=False)
            emotion_pred = model.predict(input_image, verbose=0)
            results['emotion'] = EMOTION_CLASSES[np.argmax(emotion_pred)]
            print("✅ Emotion Predicted. Unloading...")
            del model
            K.clear_session()
            gc.collect()
            sys.stdout.flush()
            
        return results

    except Exception as e:
        print(f"Face analysis error: {str(e)}")
        sys.stdout.flush()
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        if 'face_image_base64' not in request.form:
             return jsonify({'error': 'No image data'}), 400

        data_url = request.form['face_image_base64']
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        np_data = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if img is None:
             return jsonify({'error': 'Could not decode image'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_cascade_local = get_face_cascade()
        faces = face_cascade_local.detectMultiScale(gray, 1.3, 5)
        face_detected = len(faces) > 0
        eyes_open = False
        
        # Check eyes in the largest face
        if face_detected:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            roi_gray = gray[y:y+h, x:x+w]
            eye_cascade_local = get_eye_cascade()
            eyes = eye_cascade_local.detectMultiScale(roi_gray, 1.1, 3)
            # Filter minimal size to avoid noise
            eyes = [e for e in eyes if e[2]*e[3] > 100] 
            if len(eyes) >= 2:
                eyes_open = True

        return jsonify({
            'face_detected': face_detected,
            'eyes_open': eyes_open
        })

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/module/realtime')
def realtime_module():
    return render_template('realtime.html')

@app.route('/module/upload')
def upload_module():
    return render_template('upload.html')

@app.route('/report')
def report_page():
    return render_template('report.html')

@app.route('/analyze', methods=['POST'])
def analyze():

    face_image_path = None
    audio_file_path = None
    try:
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'face_analysis': {},
            'voice_analysis': {},
            'overall_condition': 'Assessment pending',
            'clinical_observations': [],
            'recommendations': []
        }

        # --- Face Analysis ---
        face_image_path = None
        
        # Check if image is uploaded as file or base64 string
        if 'face_image' in request.files:
            file = request.files['face_image']
            if file.filename != '':
                filename = secure_filename(f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                face_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(face_image_path)
        elif 'face_image_base64' in request.form:
             # Handle base64 image from webcam
             data_url = request.form['face_image_base64']
             header, encoded = data_url.split(",", 1)
             data = base64.b64decode(encoded)
             filename = secure_filename(f"face_realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
             face_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
             with open(face_image_path, "wb") as f:
                 f.write(data)

        # Get patient name
        patient_name = request.form.get('patient_name', 'Anonymous')
        results['patient_name'] = patient_name

        if face_image_path:
            # Convert image to base64 for report display (since we delete the file)
            with open(face_image_path, "rb") as image_file:
                 encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            results['face_image_path'] = f"data:image/jpeg;base64,{encoded_string}"
            
            face_results = analyze_face_image(face_image_path)
            results['face_analysis'] = face_results
        
        # --- Voice Analysis ---
        audio_file_path = None
        if 'voice_audio' in request.files:
            file = request.files['voice_audio']
            if file.filename != '':
                filename = secure_filename(f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
                audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(audio_file_path)
        
        if audio_file_path:
            results['audio_file_path'] = audio_file_path
            voice_results = analyze_voice_file(audio_file_path)
            results['voice_analysis'] = voice_results

        # --- Combine Results ---
        results['overall_condition'] = determine_overall_condition(results['face_analysis'], results['voice_analysis'])
        results['clinical_observations'] = generate_clinical_observations(results['face_analysis'], results['voice_analysis'])
        results['recommendations'] = generate_medical_recommendations(results['overall_condition'])

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup temporary files
        if face_image_path and os.path.exists(face_image_path):
            os.remove(face_image_path)
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)

def analyze_voice_file(audio_path):
    """Analyze audio file for features"""
    try:
        # Heuristic analysis based on file properties since we don't have the live stream data
        # In a real app, uses librosa or openSMILE. Here we simulate based on the user's logic.
        
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            
            # Read all frames
            frames_data = f.readframes(frames)
            audio_data = np.frombuffer(frames_data, dtype=np.int16)
            
            if len(audio_data) == 0:
                return {}

            # Calculate basic features
            volume_values = np.sqrt(np.mean(audio_data**2)) # RMS
            
            # Simple zero crossing rate for pitch approx
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
            avg_pitch = zero_crossings * rate / 2
            
            # Stability (variance)
            # Since we have one big chunk, we can't do real variance easily without chunking
            # creating pseudo chunks
            chunk_size = int(len(audio_data) / 10)
            if chunk_size > 0:
                chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]
                vol_vars = [np.sqrt(np.mean(c**2)) for c in chunks]
                vol_stability = 100 - min(100, np.std(vol_vars) / (np.mean(vol_vars) + 1) * 100)
            else:
                vol_stability = 80 # default
            
            # Scaling to visualization values
            avg_volume = min(100, volume_values / 32767 * 100 * 5) # Amplify
            
            # Logic from original code adapted
            volume_var = 100 - vol_stability
            pitch_var = 10 # simulated
            
            # --- Advanced Metrics Calculation ---
            
            # 1. Psychological Indicators
            anxiety_score = min(100, (avg_pitch / 5) + (pitch_var / 50) + (rate / 1000)) # High pitch -> Anxiety
            depression_score = min(100, max(0, (200 - avg_pitch) / 2 + (50 - avg_volume))) # Low pitch/vol -> Depression
            stress_score = min(100, volume_var / 2 + abs(avg_pitch - 200) / 10 + 20)
            stability_score = vol_stability
            
            scores = {
                'anxiety': anxiety_score,
                'depression': depression_score,
                'stress': stress_score,
                'stability': stability_score
            }
            
            # Normalization
            for k, v in scores.items():
                scores[k] = float(f"{v:.1f}")

            # 2. Dominant Emotion
            # Map weighted scores to emotion
            emo_map = {'anxiety': anxiety_score, 'depression': depression_score, 'stress': stress_score}
            primary_emotion = max(emo_map, key=emo_map.get)
            
            # Explanation
            explanations = {
                'anxiety': "Elevated pitch and rapid speech patterns detected, suggesting heightened arousal.",
                'depression': "Lower volume and pitch variance observed, consistent with subdued emotional state.",
                'stress': "Irregularities in vocal tone and pitch stability indicate potential stress markers.",
                'normal': "Vocal indicators fall within standard ranges of variation."
            }
            emotion_explanation = explanations.get(primary_emotion, "Analysis completed successfully.")
            
            # 3. Voice Characteristics
            tone = "LOUD" if avg_volume > 70 else "SOFT" if avg_volume < 30 else "MODERATE"
            pitch_desc = "HIGH" if avg_pitch > 250 else "LOW" if avg_pitch < 150 else "NORMAL"
            stability_desc = "STABLE" if stability_score > 70 else "VARIED" if stability_score > 40 else "UNSTABLE"

            # 4. Sentiment & Match (Simulated)
            # Without ASR, we simulate reading accuracy based on stability (assuming stable reading = good match)
            reading_match = min(0.98, max(0.60, stability_score / 100 + 0.1))
            sentiment = "Positive" if primary_emotion == 'normal' or (avg_pitch > 150 and avg_volume > 40) else "Neutral" if primary_emotion == 'stress' else "Negative"
            confidence = int(min(98, stability_score + 10))

            # 5. Risk & Score
            # Mental Health Score (100 = Perfect, 0 = Critical)
            # Weighted: 40% stability, 20% inv-anxiety, 20% inv-depression, 20% inv-stress
            mh_score = (stability_score * 0.4) + \
                       ((100 - anxiety_score) * 0.2) + \
                       ((100 - depression_score) * 0.2) + \
                       ((100 - stress_score) * 0.2)
            mh_score = float(f"{mh_score:.1f}")

            if mh_score >= 80: risk_level = "LOW RISK"
            elif mh_score >= 50: risk_level = "MODERATE RISK"
            else: risk_level = "HIGH RISK"

            # 6. Combined Short Explanation
            short_expl = f"The subject demonstrates {primary_emotion} markers. Voice stability is {stability_desc.lower()} ({stability_score:.1f}%). Risk analysis places the subject in the {risk_level} category."

            return {
                'primary_emotion': primary_emotion,
                'voice_features': {
                    'tone': tone,
                    'pitch': pitch_desc,
                    'speech_rate': "NORMAL",
                    'volume_level': f"{avg_volume:.1f}%",
                    'pitch_level': f"{avg_pitch:.0f} Hz",
                    'stability': stability_desc
                },
                'advanced_metrics': {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'dominant_emotion': primary_emotion.upper(),
                    'emotion_explanation': emotion_explanation,
                    'reading_accuracy': reading_match,
                    'psychological': scores,
                    'risk_level': risk_level,
                    'mh_score': mh_score,
                    'explanation': short_expl
                }
            }

    except Exception as e:
        print(f"Voice analysis error: {str(e)}")
        return {'error': str(e)}

def determine_overall_condition(face_results, voice_results):
    face_emotion = face_results.get('emotion', 'neutral') if face_results else 'neutral'
    voice_emotion = voice_results.get('primary_emotion', 'neutral') if voice_results else 'neutral'
    
    if face_emotion in ['sad', 'fear'] or voice_emotion in ['depression', 'anxiety']:
        return "Moderate emotional distress observed"
    elif face_emotion == 'angry' or voice_emotion == 'stress':
        return "Signs of stress and agitation present"
    elif face_emotion == 'happy' and voice_emotion == 'neutral':
        return "Generally stable emotional state"
    else:
        return "Normal variation in emotional expression"

def generate_clinical_observations(face_results, voice_results):
    observations = []
    if face_results:
        face_str = face_results.get('emotion', 'unknown')
        gender_str = face_results.get('gender', 'unknown')
        observations.append(f"Facial analysis indicates {face_str} expression ({gender_str}).")
    
    if voice_results:
        feat = voice_results.get('voice_features', {})
        tone = feat.get('tone', 'MODERATE')
        pitch = feat.get('pitch', 'NORMAL')
        observations.append(f"Vocal tone is {tone} with {pitch} pitch.")
        observations.append(f"Acoustic analysis suggests {voice_results.get('primary_emotion', 'unknown')}.")
        
    return observations

def generate_medical_recommendations(condition):
    recommendations = {
        "Moderate emotional distress observed": [
            "Clinical Recommendation: Follow-up with mental health professional recommended",
            "Consider cognitive behavioral therapy (CBT) sessions",
            "Practice daily mindfulness and grounding exercises",
            "Maintain regular sleep schedule (7-9 hours)"
        ],
        "Signs of stress and agitation present": [
            "Clinical Recommendation: Stress management techniques advised",
            "Practice deep breathing exercises (4-7-8 technique)",
            "Consider reducing caffeine and stimulant intake",
            "Implement regular breaks during work hours"
        ],
        "Generally stable emotional state": [
            "Clinical Recommendation: Continue current healthy practices",
            "Maintain regular social connections",
            "Continue regular exercise routine"
        ],
        "Normal variation in emotional expression": [
            "Clinical Recommendation: No immediate intervention needed",
            "Continue with healthy lifestyle habits",
            "Stay connected with support system"
        ]
    }
    return recommendations.get(condition, recommendations["Normal variation in emotional expression"])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
