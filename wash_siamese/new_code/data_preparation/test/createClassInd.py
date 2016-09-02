import os

#print [name for name in os.listdir(".") if os.path.isdir(name)]

a = open("classInd.txt", "w")
#it = 1
#b= [ a.write(str(name) + os.linesep for name in os.listdir(".") if os.path.isdir(name)]
#print (os.listdir("."))

for name in os.listdir("."):
    if os.path.isdir(name):
        a.write(str(name)+ os.linesep)
