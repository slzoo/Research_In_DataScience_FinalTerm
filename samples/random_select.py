import numpy as np
import os

# List all files in dir
files = os.listdir("binary_img")

# Select 0.7 of the files randomly 
random_files = np.random.choice(files, int(len(files)*.7), replace=False)

# Get the remaining files
other_files = [x for x in files if x not in random_files]

# Do something with the files
for x in random_files:
    # print(x)
    # print("mv binary_img/"+x+" dataset/train")
    os.system("mv binary_img/"+x+" dataset/train")

# print("mv binary_img/* dataset/test")
# os.system("mv binary_img/* dataset/test")
