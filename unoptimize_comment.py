
import streamlit as st                 # Web UI framework
import pygame                          # For audio playback
import psutil                          # To monitor CPU and memory usage
import os                              # For file operations
import matplotlib.pyplot as plt        # For plotting graphs
from pydub import AudioSegment         # For audio loading and conversion
import numpy as np                     # For numerical and signal processing

# Initialize session state variables
if 'memory_usage' not in st.session_state:
    st.session_state.memory_usage = []  # Store memory usage per time step
if 'cpu_usage' not in st.session_state:
    st.session_state.cpu_usage = []     # Store CPU usage per time step
if 'playing' not in st.session_state:
    st.session_state.playing = False    # Track whether audio is playing

# Initialize the pygame audio mixer
pygame.mixer.init()

# Define a function to get memory usage in MB
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# App title
st.title("🎧 Unoptimized RAM Music Player + CPU + Memory + Frequency Spectrum")

# Upload a .wav or .mp3 audio file
uploaded_file = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])

# Once a file is uploaded
if uploaded_file:
    # Save the uploaded file to disk
    filepath = os.path.join("./", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ File uploaded!")

    # Create playback and control buttons
    col1, col2, col3 = st.columns(3)

    # Play the uploaded audio using pygame
    if col1.button("▶️ Play"):
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            st.session_state.playing = True
        except Exception as e:
            st.error(f"Error: {e}")

    # Stop the audio playback
    if col2.button("⏹ Stop"):
        pygame.mixer.music.stop()
        st.session_state.playing = False

    # Clear all usage graphs
    if col3.button("🧹 Clear Graphs"):
        st.session_state.memory_usage = []
        st.session_state.cpu_usage = []

    # When audio is playing, process and visualize data
    if st.session_state.playing:
        try:
            # Load the audio file using pydub
            if filepath.endswith(".mp3"):
                audio = AudioSegment.from_mp3(filepath)
            else:
                audio = AudioSegment.from_wav(filepath)

            # Display mono or stereo information
            channel_info = "Mono" if audio.channels == 1 else "Stereo"
            st.info(f"🎵 This audio file is: **{channel_info}** ({audio.channels} channel{'s' if audio.channels > 1 else ''})")

            # Convert pydub audio data to NumPy array
            samples = np.array(audio.get_array_of_samples())
            
            # If stereo, reshape and average the two channels to convert to mono
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1) #this make it if it stereo they will use the default dtype of float64 in our macbook air m1 ,sample in this line is mono sample

            # Show waveform format and memory usage
            st.write("Waveform dtype:", samples.dtype)
            st.write("Waveform size (MB):", samples.nbytes / (1024 * 1024))

            frame_rate = audio.frame_rate               # Sampling rate (Hz)
            total_secs = int(len(samples) / frame_rate) # Total duration in seconds

            st.subheader("📊Unoptimzie Real-Time Waveform + CPU + Memory + Frequency Spectrum (mono graph)")
            placeholder = st.empty()  # A container to dynamically update plots

            # Loop through the audio by second
            for sec in range(total_secs):
                # Stop if music is no longer playing
                if not pygame.mixer.music.get_busy():
                    st.session_state.playing = False
                    break

                # Slice waveform to extract current 1-second segment
                start = sec * frame_rate
                end = start + frame_rate
                segment = samples[start:end]

                # Collect memory and CPU usage
                mem = get_memory_usage()
                cpu = psutil.Process(os.getpid()).cpu_percent(interval=1)
                st.session_state.memory_usage.append(mem)
                st.session_state.cpu_usage.append(cpu)

                # Compute FFT to get frequency components
                fft_data = np.fft.fft(segment)
                fft_freq = np.fft.fftfreq(len(segment), d=1 / frame_rate)
                positive_freqs = fft_freq[:len(fft_freq)//2]        # Only positive frequencies
                magnitude = np.abs(fft_data[:len(fft_data)//2])     # Magnitude spectrum

                # Create subplot with 4 graphs
                fig, axs = plt.subplots(4, 1, figsize=(6, 8))

                # Plot 1: Waveform for 1 second
                axs[0].plot(segment, color='green')
                axs[0].set_title("Waveform (1 sec)")
                axs[0].set_ylabel("Amplitude")
                axs[0].grid(True)

                # Plot 2: CPU usage over time
                axs[1].plot(st.session_state.cpu_usage, color='orange')
                axs[1].set_title("CPU Usage (%)")
                axs[1].set_ylabel("CPU %")
                axs[1].set_ylim(0, 100)
                axs[1].grid(True)

                # Plot 3: Memory usage over time
                axs[2].plot(st.session_state.memory_usage, color='blue')
                axs[2].set_title("Memory Usage (MB)")
                axs[2].set_xlabel("Time Step")
                axs[2].set_ylabel("Memory")
                axs[2].grid(True)

                # Plot 4: Frequency spectrum using FFT
                axs[3].plot(positive_freqs, magnitude, color='purple')
                axs[3].set_title("Frequency Spectrum (Hz)")
                axs[3].set_xlabel("Frequency (Hz)")
                axs[3].set_ylabel("Magnitude")
                axs[3].set_xlim(0, frame_rate // 2)
                axs[3].grid(True)

                # Display all plots
                fig.tight_layout()
                placeholder.pyplot(fig)
                plt.close(fig)  # Free up memory

        except Exception as e:
            st.error(f"❌ Error processing waveform: {e}")
