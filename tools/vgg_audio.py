import torch as pt
import os
from tqdm import tqdm
import numpy as np

audio_path = "/mnt/welles/scratch/datasets/epic-audio/"

audio_out_path = "/mnt/welles/scratch/datasets/epic-vgg/"

vggish = pt.hub.load("harritaylor/torchvggish", "vggish")
vggish.embeddings = pt.nn.Sequential(*list(vggish.embeddings.children())[:-1])
vggish.eval()


audio_files = os.listdir(audio_path)
with pt.no_grad():
    for audio_file in tqdm(audio_files):
        cur_file = os.path.join(audio_path, audio_file)
        # Define output save path
        audio_save_path = os.path.join(audio_out_path, f"{audio_file[:-4]}.npy")
        if os.path.exists(audio_save_path):
            continue
        x = vggish.forward(cur_file) 
        x = x.cpu().numpy()
        np.save(audio_save_path, x)
    