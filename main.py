# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import deeplabcut
import os

video_dir = "/Users/anacarolinabotturabarros/University of Glasgow/kohl-lab - RSCABAPP2/Experiments/coho0007_fopmt1_AB/20210308/"

video_path_list = [os.path.join(video_dir,f) for f in os.listdir(video_dir) if (os.path.isfile(os.path.join(video_dir, f)) | f.endswith(".avi"))]

config_path = deeplabcut.create_new_project('coho0007_fopmt1_AB', 'AB', video_path_list)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
