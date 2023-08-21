import json

# Load the JSON file
with open('/home/ed/actionformer_release/data/thumos/annotations/thumos14.json', 'r') as file:
    data = json.load(file)

# Load the classes.txt file and create a dictionary mapping class names to integer IDs
class_to_id = {}
with open('classes.txt', 'r') as file:
    lines = file.readlines()[1:]  # Skip the header line
    for line in lines:
        class_id, class_name = line.strip().split(',')
        class_to_id[class_name.strip('"')] = int(class_id)

# Create two empty lists for validation and test videos
validation_videos = []
test_videos = []

# Iterate through the JSON data and separate validation and test videos
for video_id, video_info in data['database'].items():
    subset = video_info['subset']
    annotations = video_info['annotations']
    if annotations:  # Check if there are annotations available
        class_name = annotations[0]['label']
        class_id = class_to_id[class_name]
        if subset == 'Validation':
            validation_videos.append(f"{video_id} {class_id}")
        elif subset == 'Test':
            test_videos.append(f"{video_id} {class_id}")

# Write the validation videos to a txt file
with open('validation_videos.txt', 'w') as file:
    for line in validation_videos:
        file.write(line + '\n')

# Write the test videos to a txt file
with open('test_videos.txt', 'w') as file:
    for line in test_videos:
        file.write(line + '\n')
