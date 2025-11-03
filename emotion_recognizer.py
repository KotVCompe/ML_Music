import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class EmotionRecognizer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.emotions = ['neutral', 'happy', 'sad', 'angry']
    
    def extract_advanced_features(self, audio, sr):
        """Расширенное извлечение признаков для эмоций"""
        features = []
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(spectral_rolloff))
        
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(tempo)
        
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        
        harmonic = librosa.effects.harmonic(audio)
        features.append(np.mean(harmonic))
        
        return np.array(features)
    
    def train_demo_model(self):
        """Создание демо-модели (в реальности нужны данные для обучения)"""
        self.model = SVC(kernel='rbf', probability=True)
        
        X_demo = np.random.randn(100, 46)  # 46 features
        y_demo = np.random.choice(self.emotions, 100)
        
        X_scaled = self.scaler.fit_transform(X_demo)
        self.model.fit(X_scaled, y_demo)
    
    def predict_emotion(self, audio_path):
        """Предсказание эмоции по аудио файлу"""
        if self.model is None:
            self.train_demo_model()
        
        audio, sr = librosa.load(audio_path, sr=22050)
        features = self.extract_advanced_features(audio, sr)
        
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probabilities

# Интеграция с Qt
class EmotionAnalysisThread(QThread):
    emotion_result = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.recognizer = EmotionRecognizer()
    
    def run(self):
        try:
            emotion, probabilities = self.recognizer.predict_emotion(self.file_path)
            
            result = f"🎭 Распознанная эмоция: {emotion}\n\n"
            result += "Вероятности:\n"
            for emo, prob in zip(self.recognizer.emotions, probabilities):
                result += f"- {emo}: {prob:.2%}\n"
            
            self.emotion_result.emit(result)
        except Exception as e:
            self.emotion_result.emit(f"❌ Ошибка: {str(e)}")