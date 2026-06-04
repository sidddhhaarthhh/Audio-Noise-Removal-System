import os
import tempfile
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt

from audio_processing import enhance_audio


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Audio Noise Removal System",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.18), transparent 35%),
            radial-gradient(circle at top right, rgba(168, 85, 247, 0.14), transparent 35%),
            linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
        border-right: 1px solid #1e293b;
    }

    [data-testid="stHeader"] {
        background: rgba(2, 6, 23, 0);
    }

    .main-title {
        font-size: 52px;
        font-weight: 900;
        line-height: 1.1;
        color: #f8fafc;
        margin-bottom: 14px;
    }

    .gradient-text {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        font-size: 18px;
        color: #cbd5e1;
        line-height: 1.7;
        max-width: 950px;
        margin-bottom: 24px;
    }

    .hero-card {
        padding: 38px;
        border-radius: 28px;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
        margin-bottom: 28px;
    }

    .glass-card {
        padding: 24px;
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.28);
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 18px;
        line-height: 1.6;
    }

    .pill {
        display: inline-block;
        padding: 8px 14px;
        margin: 4px 6px 4px 0;
        border-radius: 999px;
        background: rgba(14, 165, 233, 0.12);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.25);
        font-size: 14px;
        font-weight: 600;
    }

    .pipeline-step {
        padding: 18px;
        border-radius: 18px;
        background: rgba(2, 6, 23, 0.75);
        border: 1px solid rgba(71, 85, 105, 0.55);
        text-align: center;
        min-height: 135px;
    }

    .step-number {
        font-size: 15px;
        color: #38bdf8;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .step-title {
        font-size: 18px;
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .step-text {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.5;
    }

    .metric-card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(2, 6, 23, 0.75);
        border: 1px solid rgba(71, 85, 105, 0.55);
        text-align: center;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 22px;
        font-weight: 900;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(2, 6, 23, 0.72);
        border: 1px dashed rgba(56, 189, 248, 0.65);
        border-radius: 22px;
        padding: 18px;
    }

    .stDownloadButton button {
        background: linear-gradient(90deg, #0284c7, #7c3aed);
        color: white;
        border-radius: 14px;
        height: 52px;
        font-weight: 800;
        border: none;
    }

    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #0369a1, #6d28d9);
        color: white;
        border: none;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px 0 10px 0;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- HELPER FUNCTIONS ----------------
def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)


def plot_spectrogram(audio, sr, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    stft = librosa.stft(audio)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    img = librosa.display.specshow(
        db,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig, use_container_width=True)


def calculate_basic_stats(audio, sr):
    duration = len(audio) / sr
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    return duration, peak, rms


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎧 Audio Enhancer")
    st.caption("Noise removal and signal enhancement using DSP")

    st.markdown("---")

    st.markdown("### What this app does")
    st.write("Removes background noise, enhances clarity, and visualizes signal improvement.")

    st.markdown("### Techniques used")
    st.write("• Spectral gating")
    st.write("• Butterworth band-pass filtering")
    st.write("• Amplitude normalization")
    st.write("• STFT-based spectrogram analysis")

    st.markdown("---")

    st.markdown("### Best test audio")
    st.write("Try voice recordings with fan noise, traffic noise, static noise, or low clarity.")

    st.markdown("---")

    st.success("Built with Python + Streamlit")


# ---------------- HERO SECTION ----------------
st.markdown(
    """
    <div class="hero-card">
        <div class="main-title">
            Audio Noise Removal & <span class="gradient-text">Signal Enhancement</span>
        </div>
        <div class="subtitle">
            Upload a noisy audio file and enhance it using Digital Signal Processing.
            Compare original and enhanced signals through audio playback, waveform plots,
            and frequency-domain spectrogram analysis.
        </div>
        <span class="pill">Python</span>
        <span class="pill">Streamlit</span>
        <span class="pill">Librosa</span>
        <span class="pill">SciPy</span>
        <span class="pill">Noisereduce</span>
        <span class="pill">DSP Project</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- PIPELINE SECTION ----------------
st.markdown(
    """
    <div class="glass-card">
        <div class="section-title">Processing Pipeline</div>
        <div class="section-subtitle">
            The uploaded audio passes through a complete enhancement pipeline.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">STEP 01</div>
            <div class="step-title">Upload</div>
            <div class="step-text">User uploads a noisy WAV or MP3 audio file.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with p2:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">STEP 02</div>
            <div class="step-title">Denoise</div>
            <div class="step-text">Spectral gating suppresses background noise.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with p3:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">STEP 03</div>
            <div class="step-title">Filter</div>
            <div class="step-text">Band-pass filtering removes unwanted frequencies.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with p4:
    st.markdown(
        """
        <div class="pipeline-step">
            <div class="step-number">STEP 04</div>
            <div class="step-title">Enhance</div>
            <div class="step-text">Audio is normalized and saved as output.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# ---------------- UPLOAD SECTION ----------------
st.markdown(
    """
    <div class="glass-card">
        <div class="section-title">Upload Your Audio</div>
        <div class="section-subtitle">
            Supported formats: WAV and MP3. After upload, the system will automatically process the audio.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload noisy audio file",
    type=["wav", "mp3"],
    label_visibility="collapsed"
)


# ---------------- PROCESS AUDIO ----------------
if uploaded_file is None:
    st.info("Upload an audio file to start noise removal and signal enhancement.")
else:
    os.makedirs("outputs", exist_ok=True)

    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
        temp_input.write(uploaded_file.getvalue())
        input_path = temp_input.name

    output_path = "outputs/enhanced_audio.wav"

    try:
        with st.spinner("Enhancing audio... Please wait."):
            original_audio, enhanced_audio, sr = enhance_audio(input_path, output_path)

        st.success("Audio enhanced successfully!")

    except Exception as e:
        st.error("Something went wrong while processing the audio.")
        st.write("Error details:", e)

        if os.path.exists(input_path):
            os.remove(input_path)

        st.stop()

    original_duration, original_peak, original_rms = calculate_basic_stats(original_audio, sr)
    enhanced_duration, enhanced_peak, enhanced_rms = calculate_basic_stats(enhanced_audio, sr)

    # ---------------- METRICS ----------------
    st.markdown("## Audio Details")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">File Name</div>
                <div class="metric-value">{uploaded_file.name[:18]}...</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">File Size</div>
                <div class="metric-value">{file_size_mb:.2f} MB</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Duration</div>
                <div class="metric-value">{original_duration:.2f}s</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sample Rate</div>
                <div class="metric-value">{sr} Hz</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- AUDIO PLAYER SECTION ----------------
    st.markdown("## Before vs After Audio")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Original Audio</div>
                <div class="section-subtitle">Input signal before enhancement.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.audio(uploaded_file)

    with a2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Enhanced Audio</div>
                <div class="section-subtitle">Output signal after noise removal and enhancement.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.audio(output_path)

    # ---------------- ANALYSIS TABS ----------------
    st.markdown("## Signal Visualization")

    tab1, tab2, tab3 = st.tabs(
        ["Waveform Comparison", "Spectrogram Comparison", "Signal Stats"]
    )

    with tab1:
        w1, w2 = st.columns(2)

        with w1:
            plot_waveform(original_audio, sr, "Original Audio Waveform")

        with w2:
            plot_waveform(enhanced_audio, sr, "Enhanced Audio Waveform")

    with tab2:
        s1, s2 = st.columns(2)

        with s1:
            plot_spectrogram(original_audio, sr, "Original Audio Spectrogram")

        with s2:
            plot_spectrogram(enhanced_audio, sr, "Enhanced Audio Spectrogram")

    with tab3:
        st.markdown("### Basic Signal Statistics")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Original Signal")
            st.write(f"Peak Amplitude: `{original_peak:.4f}`")
            st.write(f"RMS Value: `{original_rms:.4f}`")
            st.write(f"Duration: `{original_duration:.2f} seconds`")

        with c2:
            st.markdown("#### Enhanced Signal")
            st.write(f"Peak Amplitude: `{enhanced_peak:.4f}`")
            st.write(f"RMS Value: `{enhanced_rms:.4f}`")
            st.write(f"Duration: `{enhanced_duration:.2f} seconds`")

    # ---------------- DOWNLOAD ----------------
    st.markdown("## Download Enhanced Output")

    with open(output_path, "rb") as file:
        st.download_button(
            label="Download Enhanced Audio",
            data=file,
            file_name="enhanced_audio.wav",
            mime="audio/wav",
            use_container_width=True
        )

    if os.path.exists(input_path):
        os.remove(input_path)


# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        Developed by Siddharth Saxena • Audio Noise Removal & Signal Enhancement System
    </div>
    """,
    unsafe_allow_html=True
)