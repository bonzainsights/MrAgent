from agents.core import AgentCore
from config.settings import NVIDIA_KEYS

# Use Llama 3.2 11B Vision Instruct
# Generate a dummy image in uploads
import os
from PIL import Image

os.makedirs("data/uploads", exist_ok=True)
img = Image.new('RGB', (100, 100), color = 'red')
img.save('data/uploads/test_red.png')

agent = AgentCore(model_mode="auto")
response = agent.chat("What color is this image? [Attached Image: data/uploads/test_red.png]", stream=False)
print("Response:", response)
