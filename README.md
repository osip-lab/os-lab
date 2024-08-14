# os-lab
This repository contains the scripts of the lab of Osip Schwartz. it does not include the specific simulations of Kali regarding p

# How to install on new computers:
If you don't have a github user already - open one.

download git from here: https://git-scm.com/download/win  (use default settings everywhere: next, next, next...)

Open the git Gui and clone the repository to a local path of your choice on your computer:
![image](https://github.com/user-attachments/assets/e4208795-88a5-402f-a09e-1adda10f5aac)
The path to the repository that is in the image should be the link to the github page of this repository: https://github.com/mkali-weizmann/os-lab

Go to pycharm and open the to which you cloned the git project as a new project:
![image](https://github.com/user-attachments/assets/a5bf101b-432a-4f65-8457-9230f352da71)

He will suggest you to open a venv enviorenment for the project. If your local folder is in a cloud service such as Dropbox or Google Drive, it is recommend that you change the suggested path in the first row to another path which sits outside of the cloud folder. The reason for this it that there will be many automatically generated files in the venv, which we don't want the cloud service to constantly upload and download we we work on the project.
![image](https://github.com/user-attachments/assets/9ef86347-4249-49fe-9d89-4387e4080bb8)

* **Note**: Do not work on the same shared folder together - if you work on the same shared folder your code will be ran over by someone else before you commited it, and changes will be lost. It is fine if your folder is in a cloud storage service, but there should be a different folder for each pesron to work with it.

In the requirements.txt file all the python packages that are required for the project are specified. Pycharm will detect them and offer to install them for you. If it failed, you can download them manually using pip. It is important that you do it through the Pycharm terminal, as you want those packages to be installed in the venv directory, and using the pycharm's terminal guarantees it.

to do so, go to the terminal window in pycharm and install the packages, specifying the version that appears in the requirements file.
Here is an example of installing a specific version of matplotlib (typing 'pip install -v "matplotlib==3.9.2"' into the terminal and then pressing enter):

![image](https://github.com/user-attachments/assets/1812c0ef-f737-472b-8eec-54a3bddaa6e7)

# How to add people to the project:
![image](https://github.com/user-attachments/assets/b39eb518-70e0-4e77-af6e-ca12223bfd2c)

# How to ignore big data files in the project (ignore = don't track and don't upload them with git):
add them to the .gitignore text file:
![image](https://github.com/user-attachments/assets/22f727dc-4804-4ad9-ba1e-fcb172bfaaf5)

I think that's it for now.

**You are welcome to add here any instructions of usage to any part of the code for other people to use as well.
**





