import os
import sys
import numpy as np

#outTrainFile = open("training_paths.txt", "w")
#outTestFile = open ("testing_paths.txt", "w")
outTrainFile = open ("training_paths_2.txt", "w")
all_files = []

def interpolate (pose_list, all_files):
    full_list = []
    for i in range(len(pose_list)-1):
        start = pose_list[i] 
        end = pose_list[i+1]
        if (abs(start - end) > 300):
             end = end + 360
        inter_nums = np.linspace(start, end, 6)
        for j in range(len(inter_nums)-1):
            if (inter_nums[j] > 360):
                inter_nums[j] = inter_nums[j] - 360    
            full_list.append(inter_nums[j])
    #print (full_list)
    
    for i in range(len(all_files)):
        print ( "%d full_list %s all_files %s "  %(i, full_list[i], all_files[i]))
    #print (len(full_list))
    #print (len(all_files))
    #for j in full_list:
    #    print ( ,j)

for path, subdirs, files in os.walk(r'./'):
    for filename in files:
        #print (filename)
        #path = path[2:]
        y = '_crop' 
        z = '.png'
    
        a = 'bowl'
        b = 'banana'
        c = 'cap'
        d = 'coffee_mug'
        e = 'food_bag'
        f = 'lemon'
        g = 'marker'
        h = 'glue_stick'
        i = 'food_cup'
        j = 'instant_noodles'
        
        #if (a in filename or b in filename or c in filename or d in filename or e in filename or f in filename or g in filename or h in filename or i in filename or j in filename):
        if (y in filename and z in filename):
            line = os.path.join(path[2:], filename)
            all_files.append(str(line))
                #print (int(line.split('/')[1].split('_')[-1] + line.split('_')[2] + line.split('_')[3] + line.split('_')[4].zfill(3)))
            #print (line) 
                #print (int(line.split('/')[1].split('_')[-1] + line.split('/')[2].split('_')[-4] + line.split('/')[2].split('_')[-3] + line.split('/')[2].split('_')[-2].zfill(3) ))
            #print (int(line.split('/')[1].split('_')[-1] + line.split('/')[2].split('_')[0].zfill(3)))

#all_files = sorted(all_files, key=lambda a:  int(a.split('/')[1].split('_')[-1] + a.split('/')[2].split('_')[-4] + a.split('/')[2].split('_')[-3] + a.split('/')[2].split('_')[-2].zfill(3) ))
all_files = sorted (all_files, key=lambda a: int(line.split('/')[1].split('_')[-1] + line.split('/')[2].split('_')[0].zfill(3)))

trainFiles = []
testFiles = []

count = 1

for files in all_files:
   # if int(files.split('/')[1].split('_')[-1]) == 2:
        #testFiles.append(files)
    #    outTestFile.write(str(files) + os.linesep)
    #else:   
        #trainFiles.append(files)
    if count >78500*1 and count < (78500*2+1):
        outTrainFile.write(str(files) + os.linesep)
    count = count + 1
    #print (files)
#print (trainFiles)
#print (testFiles)

pose_list = []
'''
for files in all_files:
    if ( (int(files.split('_')[4])-1)%5 == 0 ):
        pos_file = files.replace(files.split('_')[5], 'pose.txt')                           
        f = open(pos_file, 'r')
        #print ("%s %f" %(files, float(f.read())))
        pose_list.append(float(f.read()))
'''
#interpolate (pose_list, all_files) 
