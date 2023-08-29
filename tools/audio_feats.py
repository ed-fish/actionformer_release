import os
from moviepy.editor import VideoFileClip
import torch
import librosa
import numpy as np
import glob
from tqdm import tqdm
import librosa


feature_time_interval = 0.1333  # time interval for each feature vector in seconds
feature_stride = 4  # stride of 4 frames
video_frame_rate = 30  # ~30 fps

audio_path = "/mnt/welles/scratch/datasets/thumos/audio"
out_path = "/mnt/welles/scratch/datasets/thumos/audio_np"
audio_paths = os.listdir(audio_path)

for audio_file in tqdm(audio_paths):
    f = os.path.join(out_path, audio_file[:-4] + ".npy")
    if os.path.exists(f):
        continue
    else: 
        audio_data, sr = librosa.load(os.path.join(audio_path,audio_file), sr=16000, mono=True)
        audio_samples_per_feature = int(sr * feature_time_interval) 
        num_segments = (len(audio_data) - audio_samples_per_feature) // audio_samples_per_feature
        audio_segments = np.empty((num_segments, audio_samples_per_feature))
        
        for i in range(num_segments):
            start_idx = i * audio_samples_per_feature
            end_idx = start_idx + audio_samples_per_feature
            audio_segments[i, :] = audio_data[start_idx:end_idx]
            
        np.save(os.path.join(out_path, audio_file[:-4]) + ".npy", audio_segments)

print("all features extracted")
