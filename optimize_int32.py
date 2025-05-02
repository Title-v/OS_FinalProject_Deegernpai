
import streamlit as st                 # Web-based UI framework
import pygame                          # For playing audio files
import psutil                          # For CPU and memory monitoring
import os                              # For file system operations
import matplotlib.pyplot as plt        # For plotting graphs
from pydub import AudioSegment         # For audio manipulation and conversion
import numpy as np                     # For numerical computations
import gc                              # For garbage collection and memory management

# Initialize session states to retain data across re-runs
if 'memory_usage' not in st.session_state:
    st.session_state.memory_usage = []  # Track memory usage per time step
if 'cpu_usage' not in st.session_state:
    st.session_state.cpu_usage = []     # Track CPU usage per time step
if 'playing' not in st.session_state:
    st.session_state.playing = False    # Flag to determine playback status

# Initialize pygame's mixer for audio playback
pygame.mixer.init()

# Function to get memory usage in MB
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert from bytes to MB

# Display app title
st.title("🎵 Optimized Mono Audio Player + Frequency + Memory + CPU Viewer")

# Allow user to upload an audio file
uploaded_file = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])

if uploaded_file:
    # Save uploaded file temporarily
    filepath = os.path.join("./", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ File uploaded!")

    # Display original file size
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    st.write(f"📦 Original File Size: **{file_size:.2f} MB**")

    # Load audio using pydub
    audio = AudioSegment.from_file(filepath)
    original_channels = audio.channels
    st.info(f"🎵 Original Audio Channels: **{'Mono' if original_channels == 1 else 'Stereo'}** ({original_channels} channel{'s' if original_channels > 1 else ''})")

    # Convert stereo to mono if needed
    if original_channels > 1:
        audio = audio.set_channels(1)
        st.warning("🔄 Converted to Mono for playback.")
    else:
        st.success("✅ Already Mono, no conversion needed.")

    # Save converted audio to temp mono file
    mono_path = "temp_mono.wav"
    audio.export(mono_path, format="wav")

    # Load and display info about converted file
    mono_audio = AudioSegment.from_wav(mono_path)
    st.success(f"✅ Converted Mono Channels: **{mono_audio.channels} channel**")
    mono_size = os.path.getsize(mono_path) / (1024 * 1024)
    st.write(f"📦 Converted Mono File Size: **{mono_size:.2f} MB**")

    # Delete original file and free memory
    os.remove(filepath)
    del audio
    gc.collect()

    # Create control buttons: Play, Stop, Clear Graphs
    col1, col2, col3 = st.columns(3)
    if col1.button("▶️ Play"):
        try:
            pygame.mixer.music.load(mono_path)
            pygame.mixer.music.play()
            st.session_state.playing = True
        except Exception as e:
            st.error(f"Error: {e}")

    if col2.button("⏹ Stop"):
        pygame.mixer.music.stop()
        st.session_state.playing = False

    if col3.button("🧹 Clear Graphs"):
        st.session_state.memory_usage = []
        st.session_state.cpu_usage = []

    # While audio is playing
    if st.session_state.playing:
        try:
            # Load mono file again
            audio = AudioSegment.from_wav(mono_path)
            samples = np.array(audio.get_array_of_samples()).astype(np.int32)  # Convert to int32
            frame_rate = audio.frame_rate
            total_secs = int(len(samples) / frame_rate)

            # Display waveform format and memory footprint
            st.write("💾 Waveform dtype:", samples.dtype)
            st.write(f"💾 Waveform size in RAM: **{samples.nbytes / (1024 * 1024):.2f} MB**")

            st.subheader("🎧 Optimized Real-Time Waveform + CPU + Memory + Frequency Spectrum (mono graph)")
            placeholder = st.empty()  # For dynamic plotting updates

            # Process each second of audio
            for sec in range(total_secs):
                if not pygame.mixer.music.get_busy():  # Stop if playback ends
                    st.session_state.playing = False
                    break

                # Slice the waveform by 1 second
                start = sec * frame_rate
                end = start + frame_rate
                segment = samples[start:end]

                # Track system resource usage
                mem = get_memory_usage()
                cpu = psutil.Process(os.getpid()).cpu_percent(interval=1)

                st.session_state.memory_usage.append(mem)
                st.session_state.cpu_usage.append(cpu)

                # Perform FFT analysis
                fft_data = np.fft.fft(segment)
                fft_freq = np.fft.fftfreq(len(segment), d=1 / frame_rate)
                positive_freqs = fft_freq[:len(fft_freq) // 2]
                magnitude = np.abs(fft_data[:len(fft_data) // 2])

                # Generate plots: waveform, CPU, RAM, and frequency spectrum
                fig, axs = plt.subplots(4, 1, figsize=(6, 8))

                axs[0].plot(segment, color='green')
                axs[0].set_title("Waveform (1 sec)")
                axs[0].set_ylabel("Amplitude")
                axs[0].grid(True)

                axs[1].plot(st.session_state.cpu_usage, color='orange')
                axs[1].set_title("CPU Usage (%)")
                axs[1].set_ylabel("CPU %")
                axs[1].set_ylim(0, 100)
                axs[1].grid(True)

                axs[2].plot(st.session_state.memory_usage, color='blue')
                axs[2].set_title("Memory Usage (MB)")
                axs[2].set_xlabel("Time Step")
                axs[2].set_ylabel("Memory")
                axs[2].grid(True)

                axs[3].plot(positive_freqs, magnitude, color='purple')
                axs[3].set_title("Frequency Spectrum (Hz)")
                axs[3].set_xlabel("Frequency (Hz)")
                axs[3].set_ylabel("Magnitude")
                axs[3].set_xlim(0, frame_rate // 2)
                axs[3].grid(True)

                # Display and clean up
                fig.tight_layout()
                placeholder.pyplot(fig)
                plt.close(fig)

            # Clean up temp resources after playback ends
            del samples
            del audio
            gc.collect()
            if os.path.exists(mono_path):
                os.remove(mono_path)

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
