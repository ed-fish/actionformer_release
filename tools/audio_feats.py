import os
import librosa
import numpy as np
from tqdm import tqdm

# Initialize constants for video feature extraction
feature_time_interval = 16 / 30  # 0.5333 seconds
feature_stride_time = 4 / 30  # 0.1333 seconds

# Initialize constants for audio feature extraction
audio_sampling_rate = 16000  # 16kHz
n_mels = 64  # Number of Mel bands
hop_length = int(feature_stride_time * audio_sampling_rate)  # Align with video feature stride
n_fft = int(feature_time_interval * audio_sampling_rate)  # Align with video feature window

# Initialize file paths
audio_path = "/mnt/welles/scratch/datasets/Activity-Net/v1-3/raw_audio/"
mel_audio_out_path = "/mnt/welles/scratch/datasets/Activity-Net/v1-3/wav_whole/"

# Create output directory if it doesn't exist
if not os.path.exists(mel_audio_out_path):
    os.makedirs(mel_audio_out_path)

# Iterate through audio files and extract features
audio_files = os.listdir(audio_path)

for audio_file in tqdm(audio_files):
    # Define output save path
    mel_audio_save_path = os.path.join(mel_audio_out_path, f"{audio_file[:-4]}.npy")

    # Skip if output file already exists
    if os.path.exists(mel_audio_save_path):
        continue

    try:
        # Load audio file
        audio_data, _ = librosa.load(os.path.join(audio_path, audio_file), sr=audio_sampling_rate, mono=True)
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        continue

    # Compute Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=audio_sampling_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=125, fmax=7500)

    # Compute log Mel-spectrogram
    log_mel_spec = np.log(mel_spec + 1e-6)

    # Save features
    np.save(mel_audio_save_path, log_mel_spec)

print("Mel-log-spectrograms extracted and saved.")
