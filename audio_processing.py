import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt


def load_audio(file_path, sr=22050):
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate


def bandpass_filter(audio, sr, lowcut=300, highcut=3400, order=5):
    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sr,
        output="sos"
    )
    filtered_audio = sosfiltfilt(sos, audio)
    return filtered_audio


def normalize_audio(audio):
    max_amplitude = np.max(np.abs(audio))

    if max_amplitude == 0:
        return audio

    return audio / max_amplitude


def reduce_noise(audio, sr):
    reduced_audio = nr.reduce_noise(y=audio, sr=sr)
    return reduced_audio


def enhance_audio(input_path, output_path):
    audio, sr = load_audio(input_path)

    noise_removed = reduce_noise(audio, sr)

    filtered_audio = bandpass_filter(noise_removed, sr)

    enhanced_audio = normalize_audio(filtered_audio)

    sf.write(output_path, enhanced_audio, sr)

    return audio, enhanced_audio, sr