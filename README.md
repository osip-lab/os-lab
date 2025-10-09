# os-lab
This repository contains the scripts of the lab of Osip Schwartz.
it does not include the specific simulations: ```cavity-design``` and ```electron-laser-interaction```

## How to work with data files:
* copy the file ```local_config_template.py``` to a new file, called ```local_config.py``` and set there the relevant local paths.
* Whenever wanting to read or write data, you can access it like so:

```python
import os
from local_config import PATH_DROPBOX
import pandas as pd

specific_file_path = os.path.join(PATH_DROPBOX, r"EMNV\Ring down\270624\ring_down_curve_parameters.xlsx")

pd.read_excel(specific_file_path)
```

# How to install on new computers:
If you don't have a github user already - open one.

download git from here: https://git-scm.com/download/win  (use default settings everywhere: next, next, next...)

Open the git Gui and clone the repository to a local path of your choice on your computer:
![image](https://github.com/user-attachments/assets/e4208795-88a5-402f-a09e-1adda10f5aac)
The path to the repository that is in the image should be the link to the github page of this repository: https://github.com/mkali-weizmann/os-lab

Go to pycharm and open the folder to which you cloned the git project as a new project:
![image](https://github.com/user-attachments/assets/a5bf101b-432a-4f65-8457-9230f352da71)

He will suggest you to open a venv enviorenment for the project. If your local folder is in a cloud service such as Dropbox or Google Drive, it is recommend that you change the suggested path in the first row to another path which sits outside of the cloud folder. The reason for this it that there will be many automatically generated files in the venv, which we don't want the cloud service to constantly upload and download we we work on the project.
![image](https://github.com/user-attachments/assets/9ef86347-4249-49fe-9d89-4387e4080bb8)

* **Note**: Do not work on the same shared folder together - if you work on the same shared folder your code will be ran over by someone else before you commited it, and changes will be lost. It is fine if your folder is in a cloud storage service, but there should be a different folder for each pesron to work with it.

In the ```requirements.txt``` file all the python packages that are required for the project are specified. Pycharm will detect them and offer to install them for you. If it failed, you can download them manually using pip. It is important that you do it through the Pycharm terminal, as you want those packages to be installed in the venv directory, and using the pycharm's terminal guarantees it.

to do so, go to the terminal window in pycharm and install the packages, specifying the version that appears in the requirements file.
Here is an example of installing a specific version of matplotlib (typing ```pip install -v "matplotlib==3.9.2"``` into the terminal and then pressing enter):

![image](https://github.com/user-attachments/assets/1812c0ef-f737-472b-8eec-54a3bddaa6e7)

## How to add people to the project:
![image](https://github.com/user-attachments/assets/b39eb518-70e0-4e77-af6e-ca12223bfd2c)

## How to ignore big data files in the project (ignore = don't track and don't upload them with git):
add them to the ```.gitignore``` text file:
![image](https://github.com/user-attachments/assets/22f727dc-4804-4ad9-ba1e-fcb172bfaaf5)

## How to compress heavy videos:
1. Make sure you have ffmpeg on the computer by typing ```ffmpeg``` in the terminal (Winkey+r, type cmd, enter, and then type ffmpeg in the window).
If you don't have it, download it according to those instruction: https://phoenixnap.com/kb/ffmpeg-windows
2. run utilities/compress_videos.cmd and follow the instructions (it converts all files in the folder you give it).
3. If you did not choose to delete the original files, when compressing, but wish to delete them afterwards, run utilities/video_tools/delete_redundant_avi_files.py on the same folder.

## How to conveniently work on data files that change rapidly:
This syntax takes the file from your clipboard (if it is a file with the relevant format), or wait for a file from the relevant format to be copied.
It is convenient when you run the script, each time on a different data file (csv, video, etc.) and you want to avoid changing the path in the code each time.

```python
from utilities.media_tools.utils import wait_for_path_from_clipboard

video_path = wait_for_path_from_clipboard(filetype='video')
```

## How to use the Python-Driver-for-Thorlabs-power-meter submodule:
run in terminal:
```angular2html
# make sure the parent submodule exists locally
git submodule update --init Python-Driver-for-Thorlabs-power-meter

# switch the nested submodule to HTTPS (run inside the parent submodule)
git -C Python-Driver-for-Thorlabs-power-meter submodule set-url GlobalLogger https://github.com/Tinyblack/GlobalLogger.git
git -C Python-Driver-for-Thorlabs-power-meter submodule sync --recursive

# clean any partial clone
Remove-Item -Recurse -Force .\Python-Driver-for-Thorlabs-power-meter\GlobalLogger -ErrorAction SilentlyContinue

# now fetch everything
git submodule update --init --recursive

# from the superproject root
# Point the submodule at HTTPS instead of SSH
git submodule set-url Python-Driver-for-Thorlabs-power-meter/GlobalLogger https://github.com/Tinyblack/GlobalLogger.git

# Sync submodule config to .git/config
git submodule sync --recursive

# Clean any half-created clone (optional, if it exists)
Remove-Item -Recurse -Force .\Python-Driver-for-Thorlabs-power-meter\GlobalLogger -ErrorAction SilentlyContinue

```

**You are welcome to add here any instructions of usage to any part of the code for other people to use as well.**

## For voice commands operations (under utilities/automations):

download vosk-model-small-en-us-0.15 from https://alphacephei.com/vosk/models and put it in the folder utilities/automations/models/vosk-model-small-en-us-0.15