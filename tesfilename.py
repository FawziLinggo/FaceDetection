import os,random
path_1_meter="1-meter/"

path = path_1_meter + random.choice(os.listdir(path_1_meter))

print(path)