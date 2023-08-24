import os
from moviepy.editor import VideoFileClip
import torch
import librosa
import numpy as np
import glob
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
vggish_model = vggish_model.eval().to(device)

audio_path = "/mnt/welles/scratch/datasets/thumos/audio"
out_path = "/mnt/welles/scratch/datasets/thumos/audio_features"
audio_paths = os.listdir(audio_path)
with torch.no_grad():
    for audio_file in tqdm(audio_paths):
        audio_p = os.path.join(audio_path, audio_file)
        feats = vggish_model.forward(audio_p)
        feats = feats.cpu().numpy()
        np.save(os.path.join(out_path, audio_file[:-4] + ".npy"), feats)

print("all features extracted")
