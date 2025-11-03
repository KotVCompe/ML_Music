import sys
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTextEdit, 
                             QWidget, QProgressBar, QTabWidget, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import warnings
warnings.filterwarnings('ignore')
import sounddevice as sd
import soundfile as sf
from scipy import signal

class AudioProcessor(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    visualization_ready = pyqtSignal(object)
    
    def __init__(self, file_path, operation):
        super().__init__()
        self.file_path = file_path
        self.operation = operation
        self.audio_data = None
        self.sr = None
    
    def load_audio(self):
        """Загрузка аудио файла"""
        self.audio_data, self.sr = librosa.load(self.file_path, sr=None)
        return self.audio_data, self.sr
    
    def extract_features(self):
        """Извлечение всех возможных признаков"""
        features = {}
        
        mfcc = librosa.feature.mfcc(y=self.audio_data, sr=self.sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio_data, sr=self.sr)
        features['spectral_centroid'] = float(np.mean(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.audio_data, sr=self.sr)
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.audio_data, sr=self.sr)
        features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
        
        tempo, _ = librosa.beat.beat_track(y=self.audio_data, sr=self.sr)
        features['tempo'] = float(tempo)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(self.audio_data)
        features['zero_crossing_rate'] = float(np.mean(zero_crossing_rate))
        
        rms = librosa.feature.rms(y=self.audio_data)
        features['rms'] = float(np.mean(rms))
        
        chroma = librosa.feature.chroma_stft(y=self.audio_data, sr=self.sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        
        return features
    
    def classify_audio(self, features):
        """Классификация типа аудио"""
        zcr = features['zero_crossing_rate']
        spectral_centroid = features['spectral_centroid']
        mfcc_std_mean = float(np.mean(features['mfcc_std']))
        tempo = features['tempo']
        
        if zcr > 0.08 and spectral_centroid > 1500 and tempo < 200:
            return "Речь", "Голос человека"
        elif spectral_centroid > 800 and mfcc_std_mean > 30:
            return "Музыка", "Музыкальный фрагмент"
        elif features['rms'] < 0.005:
            return "Тишина", "Очень тихий звук"
        elif spectral_centroid < 500:
            return "Низкочастотный шум", "Глухие звуки"
        else:
            return "Сложный звук", "Смешанные характеристики"
    
    def detect_emotion(self, features):
        """Детекция эмоций по голосу (упрощенная)"""
        spectral_centroid = features['spectral_centroid']
        tempo = features['tempo']
        zcr = features['zero_crossing_rate']
        rms = features['rms']
        
        if spectral_centroid > 2500 and tempo > 120 and rms > 0.05:
            return "Радость/Возбуждение", "Высокая энергия, быстрый темп"
        elif spectral_centroid < 1500 and tempo < 90 and rms < 0.03:
            return "Грусть/Усталость", "Низкая энергия, медленный темп"
        elif zcr > 0.12 and spectral_centroid > 3000 and rms > 0.08:
            return "Злость/Напряжение", "Резкий, высокочастотный звук"
        elif 0.06 < zcr < 0.09 and 1500 < spectral_centroid < 2500:
            return "Нейтрально/Спокойно", "Сбалансированные характеристики"
        else:
            return "Смешанные эмоции", "Разнообразные характеристики"
    
    def reduce_noise_simple(self):
        """Простое подавление шума с помощью фильтров"""
        nyquist = self.sr / 2
        cutoff = 8000  # Hz
        normal_cutoff = cutoff / nyquist
        
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        filtered_audio = signal.filtfilt(b, a, self.audio_data)
        
        return filtered_audio
    
    def apply_equalizer(self, low_gain=1.0, mid_gain=1.0, high_gain=1.0):
        """Простой эквалайзер"""
        # Низкие частоты (0-300 Hz)
        b_low, a_low = signal.butter(3, 300/(self.sr/2), btype='low')
        low_freq = signal.filtfilt(b_low, a_low, self.audio_data) * low_gain
        
        # Средние частоты (300-3000 Hz)
        b_mid, a_mid = signal.butter(3, [300/(self.sr/2), 3000/(self.sr/2)], btype='band')
        mid_freq = signal.filtfilt(b_mid, a_mid, self.audio_data) * mid_gain
        
        # Высокие частоты (3000+ Hz)
        b_high, a_high = signal.butter(3, 3000/(self.sr/2), btype='high')
        high_freq = signal.filtfilt(b_high, a_high, self.audio_data) * high_gain
        
        # Комбинация с нормализацией
        combined = low_freq + mid_freq + high_freq
        return np.clip(combined, -1.0, 1.0)
    
    def generate_spectrogram(self):
        """Генерация спектрограммы"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Waveform
        times = np.arange(len(self.audio_data)) / self.sr
        axes[0].plot(times, self.audio_data)
        axes[0].set_title('Волновая форма')
        axes[0].set_ylabel('Амплитуда')
        axes[0].grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
        img = librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz', 
                                      ax=axes[1], cmap='viridis')
        axes[1].set_title('Спектрограмма')
        axes[1].set_ylabel('Частота (Hz)')
        plt.colorbar(img, ax=axes[1])
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=self.audio_data, sr=self.sr, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=self.sr, x_axis='time', ax=axes[2], cmap='coolwarm')
        axes[2].set_title('🎵 MFCC (Mel-Frequency Cepstral Coefficients)')
        axes[2].set_ylabel('Коэффициенты MFCC')
        axes[2].set_xlabel('Время (секунды)')
        plt.colorbar(img, ax=axes[2])
        
        plt.tight_layout()
        return fig
    
    def run(self):
        try:
            results = {}
            
            if self.operation == "analyze":
                self.progress_updated.emit(20)
                self.load_audio()
                
                self.progress_updated.emit(50)
                features = self.extract_features()
                
                self.progress_updated.emit(70)
                audio_type, type_desc = self.classify_audio(features)
                emotion, emotion_desc = self.detect_emotion(features)
                
                self.progress_updated.emit(90)
                results = {
                    'type': 'analysis',
                    'audio_type': audio_type,
                    'type_description': type_desc,
                    'emotion': emotion,
                    'emotion_description': emotion_desc,
                    'features': features,
                    'duration': len(self.audio_data) / self.sr,
                    'sr': self.sr
                }
                
            elif self.operation == "visualize":
                self.load_audio()
                fig = self.generate_spectrogram()
                results = {'type': 'visualization', 'figure': fig}
                
            elif self.operation == "denoise":
                self.load_audio()
                cleaned_audio = self.reduce_noise_simple()
                results = {
                    'type': 'denoise', 
                    'cleaned_audio': cleaned_audio,
                    'sr': self.sr
                }
                
            elif self.operation == "equalize":
                self.load_audio()
                equalized_audio = self.apply_equalizer(1.2, 1.1, 0.9)
                results = {
                    'type': 'equalize',
                    'equalized_audio': equalized_audio,
                    'sr': self.sr
                }
            
            self.progress_updated.emit(100)
            self.result_ready.emit(results)
            
        except Exception as e:
            self.result_ready.emit({'type': 'error', 'message': str(e)})

class AudioMLWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("ML музыка")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        
        title = QLabel("ML музыка")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 15px; color: #2E86AB;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        file_layout = QHBoxLayout()
        self.select_btn = QPushButton("Загрузить аудио файл")
        self.select_btn.clicked.connect(self.select_file)
        self.select_btn.setStyleSheet(self.get_button_style("#2E86AB"))
        
        self.play_btn = QPushButton("▶Воспроизвести")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(self.get_button_style("#18A558"))
        
        file_layout.addWidget(self.select_btn)
        file_layout.addWidget(self.play_btn)
        main_layout.addLayout(file_layout)
        
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet("color: #666; margin: 5px;")
        main_layout.addWidget(self.file_label)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)
        
        self.tabs = QTabWidget()
        
        self.analysis_tab = self.create_analysis_tab()
        self.tabs.addTab(self.analysis_tab, "Анализ")
        
        self.viz_tab = self.create_visualization_tab()
        self.tabs.addTab(self.viz_tab, "Визуализация")
        
        self.processing_tab = self.create_processing_tab()
        self.tabs.addTab(self.processing_tab, "Обработка")
        
        main_layout.addWidget(self.tabs)
        
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("Результаты появятся здесь...")
        main_layout.addWidget(self.results_text)
        
        central_widget.setLayout(main_layout)
    
    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 12px 20px;
                font-size: 14px;
                border-radius: 8px;
                margin: 5px;
                min-width: 150px;
            }}
            QPushButton:hover {{
                background-color: #1A5F7A;
            }}
            QPushButton:disabled {{
                background-color: #CCCCCC;
            }}
        """
    
    def create_analysis_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        group = QGroupBox("Анализ аудио")
        group_layout = QVBoxLayout()
        
        self.analyze_btn = QPushButton("Полный анализ")
        self.analyze_btn.clicked.connect(lambda: self.process_audio("analyze"))
        self.analyze_btn.setStyleSheet(self.get_button_style("#F18F01"))
        group_layout.addWidget(self.analyze_btn)
        
        info = QLabel("Определяет тип звука, эмоции и извлекает аудио-признаки")
        info.setStyleSheet("color: #666; font-size: 12px; margin: 10px;")
        group_layout.addWidget(info)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        widget.setLayout(layout)
        return widget
    
    def create_visualization_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.viz_btn = QPushButton("Показать визуализации")
        self.viz_btn.clicked.connect(lambda: self.process_audio("visualize"))
        self.viz_btn.setStyleSheet(self.get_button_style("#C73E1D"))
        layout.addWidget(self.viz_btn)
        
        info = QLabel("Строит волновую форму, спектрограмму и MFCC")
        info.setStyleSheet("color: #666; font-size: 12px; margin: 10px;")
        layout.addWidget(info)
        
        self.viz_layout = QVBoxLayout()
        layout.addLayout(self.viz_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_processing_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Denoising
        denoise_group = QGroupBox("Шумоподавление")
        denoise_layout = QVBoxLayout()
        self.denoise_btn = QPushButton("Подавить шум")
        self.denoise_btn.clicked.connect(lambda: self.process_audio("denoise"))
        self.denoise_btn.setStyleSheet(self.get_button_style("#3F88C5"))
        denoise_layout.addWidget(self.denoise_btn)
        
        denoise_info = QLabel("Убирает высокочастотный шум с помощью фильтров")
        denoise_info.setStyleSheet("color: #666; font-size: 12px; margin: 5px;")
        denoise_layout.addWidget(denoise_info)
        
        denoise_group.setLayout(denoise_layout)
        layout.addWidget(denoise_group)
        
        # Equalizer
        eq_group = QGroupBox("Эквалайзер")
        eq_layout = QVBoxLayout()
        self.eq_btn = QPushButton("Применить эквалайзер")
        self.eq_btn.clicked.connect(lambda: self.process_audio("equalize"))
        self.eq_btn.setStyleSheet(self.get_button_style("#44BBA4"))
        eq_layout.addWidget(self.eq_btn)
        
        eq_info = QLabel("Усиливает низкие и средние частоты")
        eq_info.setStyleSheet("color: #666; font-size: 12px; margin: 5px;")
        eq_layout.addWidget(eq_info)
        
        eq_group.setLayout(eq_layout)
        layout.addWidget(eq_group)
        
        widget.setLayout(layout)
        return widget
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите аудио файл", 
            "", 
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg)"
        )
        
        if file_path:
            self.current_file = file_path
            self.play_btn.setEnabled(True)
            filename = os.path.basename(file_path)
            self.file_label.setText(f"Загружен: {filename}")
            self.results_text.setText(f"Файл '{filename}' готов к анализу!\n\nВыберите действие во вкладках выше.")
    
    def play_audio(self):
        if self.current_file:
            try:
                audio_data, sr = librosa.load(self.current_file, sr=None)
                sd.play(audio_data, sr)
                self.results_text.setText("Воспроизведение...")
            except Exception as e:
                self.results_text.setText(f"Ошибка воспроизведения: {str(e)}")
    
    def process_audio(self, operation):
        if not self.current_file:
            self.results_text.setText("Сначала загрузите аудио файл!")
            return
        
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.results_text.setText("Обработка...")
        
        self.processor = AudioProcessor(self.current_file, operation)
        self.processor.progress_updated.connect(self.progress.setValue)
        self.processor.result_ready.connect(self.handle_results)
        self.processor.start()
    
    def handle_results(self, results):
        self.progress.setVisible(False)
        
        if results['type'] == 'error':
            self.results_text.setText(f"Ошибка: {results['message']}")
            return
        
        if results['type'] == 'analysis':
            features = results['features']
            
            spectral_centroid = float(features['spectral_centroid'])
            tempo = float(features['tempo'])
            zcr = float(features['zero_crossing_rate'])
            rms = float(features['rms'])
            duration = float(results['duration'])
            sr = int(results['sr'])
            
            text = f"""
РЕЗУЛЬТАТЫ АНАЛИЗА:

Тип аудио: {results['audio_type']}
   {results['type_description']}

Эмоциональная окраска: {results['emotion']}
   {results['emotion_description']}

Технические характеристики:
   Длительность: {duration:.2f} сек
   Частота дискретизации: {sr} Гц
   Spectral Centroid: {spectral_centroid:.2f} Hz
   Tempo: {tempo:.1f} BPM
   Zero Crossing Rate: {zcr:.4f}
   RMS Energy: {rms:.4f}

Аудио-признаки извлечены успешно!
"""
            self.results_text.setText(text)
        
        elif results['type'] == 'visualization':
            for i in reversed(range(self.viz_layout.count())): 
                widget = self.viz_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            
            canvas = FigureCanvas(results['figure'])
            self.viz_layout.addWidget(canvas)
            self.results_text.setText("Визуализации сгенерированы! Посмотрите во вкладке 'Визуализация'")
        
        elif results['type'] == 'denoise':
            base_name = os.path.splitext(self.current_file)[0]
            output_path = f"{base_name}_cleaned.wav"
            sf.write(output_path, results['cleaned_audio'], int(results['sr']))
            self.results_text.setText(f"""
ШУМОПОДАВЛЕНИЕ ЗАВЕРШЕНО!

Очищенный файл сохранен как: {os.path.basename(output_path)}

Исходный файл сохранен, новый файл создан с суффиксом '_cleaned'
Вы можете воспроизвести оба файла для сравнения!
""")
        
        elif results['type'] == 'equalize':
            base_name = os.path.splitext(self.current_file)[0]
            output_path = f"{base_name}_equalized.wav"
            sf.write(output_path, results['equalized_audio'], int(results['sr']))
            self.results_text.setText(f"""
ЭКВАЛАЙЗЕР ПРИМЕНЕН!

Обработанный файл сохранен как: {os.path.basename(output_path)}

Настройки эквалайзера:
   - Низкие частоты: +20%
   - Средние частоты: +10% 
   - Высокие частоты: -10%

Сравните оригинал и обработанную версию!
""")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioMLWorkbench()
    window.show()
    sys.exit(app.exec_())