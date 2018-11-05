import imageio
import glob
import os
import argparse
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

parser = argparse.ArgumentParser(description='PNG to GIF')
parser.add_argument('--path', type=str, default='./result/nyc_taxi',
                    help='file path ')

args = parser.parse_args()


images = []
filenames = glob.glob(args.path+'/*')

filenames.sort(key=os.path.getmtime)
#filenames.sort(key=alphanum_key)
#print(filenames)
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(args.path+'/fig.gif',images,duration=0.2)
