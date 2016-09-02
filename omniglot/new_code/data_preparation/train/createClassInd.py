import os
import sys 

outTrainFile = open("classInd.txt", "w")

count = 1
for path, subdirs, files in os.walk(r'./'):
    for filename in files:
        i = '.png'
        if (i in filename):
            line = os.path.join(path[2:], filename)
            line = line.split('/')[0:2]
            full_name = str(line[0])  + str(line[1])
            if count == 1:
                buf = full_name
                outTrainFile.write(full_name + os.linesep)  
            else:
                if buf != full_name:     
                    outTrainFile.write(full_name + os.linesep)  
                    buf = full_name
            count = count + 1           
