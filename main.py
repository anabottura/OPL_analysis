# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import deeplabcut
import os

poses_dir = "/Users/anacarolinabotturabarros/PycharmProjects/OPL/pose_files/20210308/"

poses_list = [os.path.join(poses_dir, f) for f in os.listdir(poses_dir) if
              (f.endswith(".csv"))]

print(poses_list)


# OPL experiment class

# mouse class with sex, dob?, files (dataframe with date as name)

class Mouse:

    def __init__(self, mouse_id=None, pose_files=None):
        self.id = mouse_id
        self.sex = 'Undefined'
        self.data = pose_files
        self.group = None

    def get_dates(self):
        pass
