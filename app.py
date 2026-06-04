import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import os

from audio_processing import enhance_audio


st.set_page_config(
    page_title="Audio Noise Removal System",
    layout="wide"
)

st.title("Audio Noise Removal & Signal Enhancement System")

st.write(
    "Upload a noisy audio file. The system will remove background noise, "
    "enhance the signal, and generate a cleaner output audio."
)

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.subheader("Original Audio")
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    output_path = "outputs/enhanced_audio.wav"

    original_audio, enhanced_audio, sr = enhance_audio(input_path, output_path)

    st.subheader("Enhanced Audio")
    st.audio(output_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(original_audio, sr=sr, ax=ax)
        ax.set_title("Original Audio Waveform")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    with col2:
        st.subheader("Enhanced Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(enhanced_audio, sr=sr, ax=ax)
        ax.set_title("Enhanced Audio Waveform")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Original Spectrogram")
        fig, ax = plt.subplots()
        D = librosa.amplitude_to_db(abs(librosa.stft(original_audio)), ref=max)
        img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title("Original Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

    with col4:
        st.subheader("Enhanced Spectrogram")
        fig, ax = plt.subplots()
        D = librosa.amplitude_to_db(abs(librosa.stft(enhanced_audio)), ref=max)
        img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title("Enhanced Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

    with open(output_path, "rb") as file:
        st.download_button(
            label="Download Enhanced Audio",
            data=file,
            file_name="enhanced_audio.wav",
            mime="audio/wav"
        )

    os.remove(input_path)
    