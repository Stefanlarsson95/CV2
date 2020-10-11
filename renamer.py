import os
import glob

for file in glob.glob('capture/right/*.jpg'):
    path_old = file
    path_new = path_old[:14] + path_old[-10:]
    print('{}->{}'.format(path_old, path_new))
    os.rename(path_old, path_new)
