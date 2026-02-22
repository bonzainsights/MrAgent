import re
import json

message = "Hello [Attached Image: ./uploads/tests_1.png] World"
image_matches = re.findall(r'\[Attached Image: (.*?)\]', message)
print(image_matches)

text_only = re.sub(r'\[Attached Image: .*?\]', '', message).strip()
print(text_only)
