from PIL import Image
import os

img_path = "2026/projects/counter_strategy/report/score_distributions.png"
img = Image.open(img_path)
rgb_im = img.convert('RGB')
rgb_im.save(img_path.replace(".png", ".jpg"), quality=95)
print("Converted to JPG")
