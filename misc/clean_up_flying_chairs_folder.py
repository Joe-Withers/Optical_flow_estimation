
import os
import glob

# Get a list of all the file paths that ends with .txt from in specified directory
fileList1 = glob.glob('D:/Joe/Documents/University/Year 4/ResearchProject/Github/Research-Project/data/FlyingChairs2/train/*-mb_*')
fileList2 = glob.glob('D:/Joe/Documents/University/Year 4/ResearchProject/Github/Research-Project/data/FlyingChairs2/train/*-occ_*')
fileList3 = glob.glob('D:/Joe/Documents/University/Year 4/ResearchProject/Github/Research-Project/data/FlyingChairs2/train/*-oids_*')
fileList4 = glob.glob('D:/Joe/Documents/University/Year 4/ResearchProject/Github/Research-Project/data/FlyingChairs2/train/*-flow_10.flo')
# Iterate over the list of filepaths & remove each file.
i = 0
for filePath in fileList4:

    if i%1000 == 0:
        print('still running...', i)
    i=i+1
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)
