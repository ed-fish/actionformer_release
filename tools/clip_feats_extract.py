import os
import numpy as np
import torch
from torchvision.io import read_video
from transformers import CLIPProcessor, CLIPModel

# Set up the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Set up the directory paths

input_dir = "/mnt/welles/scratch/datasets/thumos/val/"
output_dir = "/mnt/welles/scratch/datasets/thumos/val_clip"
os.makedirs(output_dir, exist_ok=True)

# Parameters
frame_rate = 30  # FPS
clip_length = 16  # Number of frames per clip
stride = 4  # Stride between clips

# Placeholder text input for the model
dummy_text = ["a photo of an object"]

# Function to process video clips
def process_video(video_path):
    video, audio, info = read_video(video_path, pts_unit="sec")
    num_frames = video.shape[0]
    
    visual_features = []
    for i in range(0, num_frames - clip_length + 1, stride):
        frames = video[i:i + clip_length]
        
        # Use Hugging Face's transform
        inputs = processor(text=dummy_text, images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the device

        visual_features.append(model(**inputs).visual_embeds.mean(dim=0).detach().cpu().numpy())
    
    return np.array(visual_features)

# Process each video in the input directory
for video_file in os.listdir(input_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(input_dir, video_file)
        features = process_video(video_path)
        
        output_filename = os.path.splitext(video_file)[0] + ".npy"
        output_path = os.path.join(output_dir, output_filename)
        
        np.save(output_path, features)
        print(f"Saved features for {video_file} to {output_filename}")

print("Feature extraction complete.")

