# Audio Noise Removal & Signal Enhancement System

This project is a Python-based audio processing system that removes background noise from audio signals and enhances speech clarity using Digital Signal Processing techniques.

## Project Overview

The system allows users to upload a noisy audio file, process it using noise reduction and filtering techniques, and download the enhanced audio output. It also displays waveform and spectrogram comparisons between the original and enhanced audio.

## Features

- Upload noisy `.wav` or `.mp3` audio files
- Remove background noise using spectral gating
- Apply Butterworth band-pass filtering
- Normalize audio amplitude
- Display original and enhanced audio
- Show waveform comparison
- Show spectrogram comparison
- Download enhanced audio output
- Simple Streamlit-based web interface

## Tech Stack

- Python
- Streamlit
- Librosa
- NumPy
- SciPy
- SoundFile
- Noisereduce
- Matplotlib

## How It Works

1. The audio file is loaded using Librosa.
2. Background noise is reduced using spectral gating.
3. A Butterworth band-pass filter is applied to remove unwanted low-frequency and high-frequency noise.
4. The audio amplitude is normalized.
5. The enhanced audio is saved and displayed.
6. Waveform and spectrogram plots are generated for comparison.

## Signal Processing Concepts Used

### 1. Spectral Gating

Spectral gating reduces unwanted background noise by analyzing the frequency content of the audio signal and suppressing noise-dominant frequency components.

### 2. Band-Pass Filtering

A Butterworth band-pass filter is used to preserve useful speech frequencies while reducing unwanted low-frequency and high-frequency components.

### 3. Amplitude Normalization

Amplitude normalization adjusts the signal level so that the enhanced audio has balanced loudness.

### 4. Short-Time Fourier Transform

STFT is used to convert the audio signal from the time domain to the frequency domain for spectrogram visualization.

## Folder Structure

```text
Audio-Noise-Removal-System/
│
├── app.py
├── audio_processing.py
├── requirements.txt
├── README.md
├── .gitignore
├── samples/
├── outputs/
└── screenshots/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sidddhhaarthhh/Audio-Noise-Removal-System.git
```

Move into the project folder:

```bash
cd Audio-Noise-Removal-System
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
python -m streamlit run app.py
```

## Applications

- Speech enhancement
- Podcast audio cleaning
- Voice note improvement
- Audio preprocessing for speech recognition
- Noise reduction in recorded lectures
- Digital Signal Processing demonstration

## Future Improvements

- Add real-time microphone recording
- Add noise reduction strength slider
- Add SNR improvement calculation
- Add support for more audio formats
- Deploy the app using Streamlit Community Cloud

## Author

**Siddharth Saxena**

GitHub: [sidddhhaarthhh](https://github.com/sidddhhaarthhh)