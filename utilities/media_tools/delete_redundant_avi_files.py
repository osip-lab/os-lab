import os
from utilities.media_tools.utils import delete_redundant_avi_files


directory = r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Lab's Dropbox\Laser Phase Plate\Experiments\Results"
for root, dirs, files in os.walk(directory):
    delete_redundant_avi_files(root)
