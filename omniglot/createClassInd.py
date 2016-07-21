import os

a = open("classInd.txt", "w")

for name in os.listdir("."):
    if os.path.isdir(name):
        a.write(str(name)+ os.linesep)
