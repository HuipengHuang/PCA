import requests
import os
file_url = "https://github.com/dlaptev/RobustPCA/blob/master/examples/RobustPCA_video_demo.avi?raw=true"
output_file = "RobustPCA_video_demo.avi"

if not os.path.exists(output_file):
    print(f"Downloading {output_file}...")
    r = requests.get(file_url, stream=True)
    with open(output_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")