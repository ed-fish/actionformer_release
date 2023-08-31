# import os
# from moviepy.editor import VideoFileClip
# import torch
# import librosa
# import numpy as np
# import glob
# from tqdm import tqdm
# import librosa


# feature_time_interval = 0.1333  # time interval for each feature vector in seconds
# feature_stride = 4  # stride of 4 frames
# video_frame_rate = 30  # ~30 fps

# audio_path = "/mnt/welles/scratch/datasets/thumos/audio"
# out_path = "/mnt/welles/scratch/datasets/thumos/audio_np"
# audio_paths = os.listdir(audio_path)

# for audio_file in tqdm(audio_paths):
#     f = os.path.join(out_path, audio_file[:-4] + ".npy")
#     if os.path.exists(f):
#         continue
#     else: 
#         audio_data, sr = librosa.load(os.path.join(audio_path,audio_file), sr=16000, mono=True)
#         audio_samples_per_feature = int(sr * feature_time_interval) 
#         num_segments = (len(audio_data) - audio_samples_per_feature) // audio_samples_per_feature
#         audio_segments = np.empty((num_segments, audio_samples_per_feature))
        
#         for i in range(num_segments):
#             start_idx = i * audio_samples_per_feature
#             end_idx = start_idx + audio_samples_per_feature
#             audio_segments[i, :] = audio_data[start_idx:end_idx]
            
#         np.save(os.path.join(out_path, audio_file[:-4]) + ".npy", audio_segments)

# print("all features extracted")


import os
import librosa
import numpy as np
from tqdm import tqdm

feature_time_interval = 16 / 30  # Time interval for each video feature vector in seconds, based on 16 frames at ~30 fps
feature_stride_time = 4 / 30  # Stride time in seconds, based on 4 frames at ~30 fps

audio_sampling_rate = 16000  # 16kHz
n_fft = int(0.025 * audio_sampling_rate)  # 25 ms window
hop_length = int(0.01 * audio_sampling_rate)  # 10 ms hop
n_mels = 128  # Number of mel bands

audio_path = "/mnt/welles/scratch/datasets/thumos/audio"
mel_audio_out_path = "/mnt/welles/scratch/datasets/thumos/log_mel_audio_np"

if not os.path.exists(mel_audio_out_path):
    os.makedirs(mel_audio_out_path)

audio_paths = os.listdir(audio_path)

for audio_file in tqdm(audio_paths):
    mel_audio_save_path = os.path.join(mel_audio_out_path, audio_file[:-4] + ".npy")

    if os.path.exists(mel_audio_save_path):
        continue

    try:
        audio_data, _ = librosa.load(os.path.join(audio_path, audio_file), sr=audio_sampling_rate, mono=True)
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        continue

    audio_samples_per_feature = int(audio_sampling_rate * feature_time_interval)
    audio_samples_per_stride = int(audio_sampling_rate * feature_stride_time)

    num_segments = (len(audio_data) - audio_samples_per_feature) // audio_samples_per_stride + 1

    mel_audio_segments = np.empty((num_segments, n_mels))

    for i in range(num_segments):
        start_idx = i * audio_samples_per_stride
        end_idx = start_idx + audio_samples_per_feature

        segment = audio_data[start_idx:end_idx]
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=audio_sampling_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=125, fmax=7500)
        log_mel_spec = np.log(np.mean(mel_spec, axis=1) + 0.01)
        mel_audio_segments[i, :] = log_mel_spec

    np.save(mel_audio_save_path, mel_audio_segments)

print("All features extracted")

