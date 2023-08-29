from moviepy.editor import VideoFileClip
import os

def extract_audio(video_path, audio_folder):
    video_filename = os.path.basename(video_path)
    audio_filename = os.path.splitext(video_filename)[0] + ".wav"
    audio_path = os.path.join(audio_folder, audio_filename)
    
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(audio_path)
    
    clip.close()

def main():
    video_folder = "/mnt/welles/scratch/datasets/thumos/thumos_all/"
    audio_folder = "/mnt/welles/scratch/datasets/thumos/audio/"

    # Create the audio folder if it doesn't exist
    os.makedirs(audio_folder, exist_ok=True)

    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                extract_audio(video_path, audio_folder)
                print(f"Extracted audio from {file}")

if __name__ == "__main__":
    main()
