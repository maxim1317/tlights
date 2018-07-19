from os import listdir
from os.path import isfile, join


import logic as l

if __name__ == "__main__":

    path = '/home/oberon/vidz/train/'

    file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'mp4']
    
    for fname in file_list:
        l.init_search(path + fname, fname)

