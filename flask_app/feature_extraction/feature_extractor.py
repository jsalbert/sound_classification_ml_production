import json
import librosa
import numpy as np


class FeatureExtractor:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = json.load(file)

        self.mode=self.config['mode']
        self.sample_rate=self.config['sample_rate']
        self.n_fft=self.config['n_fft']
        self.hop_length=self.config['hop_length']
        self.n_mfcc=self.config['n_mfcc']
        self.deltas=self.config['deltas']
        self.max_padding=self.config['max_padding']

    def extract_features(self, filepath):
        # Load 4 seconds of audio (as our model has been trained only on 4s samples)
        audio_file, sample_rate = librosa.load(filepath, sr=self.sample_rate, res_type='kaiser_fast', duration=4)
        if self.mode == 'mfcc':
            audio_features = self.compute_mfcc(audio_file, self.sample_rate, self.n_fft, self.hop_length, self.n_mfcc, self.deltas)
        elif self.mode == 'stft':
            audio_features = self.compute_stft(audio_file, self.sample_rate, self.n_fft, self.hop_length)
        elif self.mode == 'mel-spectogram':
            audio_features = self.compute_mel_spectogram(audio_file, self.sample_rate, self.n_fft, self.hop_length)

        audio_features = np.pad(audio_features,
                                pad_width=((0, 0), (0, self.max_padding - audio_features.shape[1])))
        audio_features = np.expand_dims(audio_features, -1)
        return audio_features

    @staticmethod
    def compute_mel_spectogram(audio_file, sample_rate, n_fft, hop_length):
        return librosa.feature.melspectrogram(audio_file,
                                              sr=sample_rate,
                                              n_fft=n_fft,
                                              hop_length=hop_length)

    @staticmethod
    def compute_stft(audio_file, sample_rate, n_fft, hop_length):
        return librosa.stft(audio_file, n_fft=n_fft, hop_length=hop_length)

    @staticmethod
    def compute_mfcc(audio_file, sample_rate, n_fft, hop_length, n_mfcc, deltas=False):
        mfccs = librosa.feature.mfcc(audio_file,
                                    sr=sample_rate,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    )
        # Change mode from interpolation to nearest
        if deltas:
          delta_mfccs = librosa.feature.delta(mfccs, mode='nearest')
          delta2_mfccs = librosa.feature.delta(mfccs, order=2, mode='nearest')
          return np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        return mfccs
