import os

import numpy as np
from PIL import Image
import ffmpeg

# Define 16 distinctive colors
colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 165, 0),   # Orange
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (128, 128, 0),   # Olive
    (255, 192, 203), # Pink
    (128, 128, 128), # Gray
    (0, 0, 0),       # Black
    (255, 255, 255), # White
    (165, 42, 42),   # Brown
    (75, 0, 130)     # Indigo
]

os.makedirs('output', exist_ok=True)
for i, color in enumerate(colors):
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[:] = color
    Image.fromarray(img).save(f'./output/frame_{i:02d}.png')


# Create 16 images, each with a different color
# Use FFmpeg to combine the images into a video
os.remove('./assets/output.mp4')
(
    ffmpeg
    .input('./output/frame_%02d.png', framerate=1, start_number=0)
    .output('./assets/output.mp4', vcodec='libx264', pix_fmt='yuv420p', vf='fps=2')
    .run()
)

print("Video created successfully.")
