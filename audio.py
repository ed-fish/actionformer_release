from moviepy.editor import VideoFileClip
import os

def extract_audio(video_path, audio_folder):
    video_filename = os.path.basename(video_path)
    audio_filename = os.path.splitext(video_filename)[0] + ".wav"
    audio_path = os.path.join(audio_folder, audio_filename)
    if os.path.exists(audio_path):
        return
    try:
    
        clip = VideoFileClip(video_path)
        audio = clip.audio
        audio.write_audiofile(audio_path)
        
        clip.close()
    except:
        return
    
        
def main():
    # video_folder = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/raw_vids/"
    audio_folder = "/mnt/welles/scratch/datasets/epic-audio/"
    extract_audio("/home/ed/actionformer_release/P01_04.MP4", audio_folder)


    # # Create the audio folder if it doesn't exist
    # os.makedirs(audio_folder, exist_ok=True)

    # for root, _, files in os.walk(video_folder):
    #     for file in files:
    #         if file.lower().endswith('.mp4'):
    #             video_path = os.path.join(root, file)
            
    #             extract_audio(video_path, audio_folder)
    #             print(f"Extracted audio from {file}")

if __name__ == "__main__":
    main()
