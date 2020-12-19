import os
import binascii

input_dir = "binary"

for root, directories, files in os.walk(input_dir):
    for filename in files:
        filepath = os.path.join(root, filename)

        os.system("truncate -s -4 " + filepath)
