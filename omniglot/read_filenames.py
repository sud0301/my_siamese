import os
import sys

outTrainFile = open("training_paths.txt", "w")
outTestFile = open ("testing_paths.txt", "w")

for path, subdirs, files in os.walk(r'./'):
    for filename in files:
        #path = path[2:]
        a = '_13'
        b = '_14'
        c = '_15'
        d = '_16'
        e = '_17'
        f = '_18'
        g = '_19'
        h = '_20'
        i = '.png'
        if (a in filename or b in filename or c in filename or d in filename or e in filename or f in filename or g in filename or h in filename):
            if (i in filename):
                line = os.path.join(path[2:], filename)
                outTestFile.write(str(line) + os.linesep) 
        else:
            if (i in filename):
                line = os.path.join(path[2:], filename)
                outTrainFile.write(str(line) + os.linesep)
         
    
