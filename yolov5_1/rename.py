import os
# path = "home/airlab/Desktop/mirror20220614/20220614/backlight1"
path = "/home/airlab/Desktop/mirror20220614/20220614/blue-backlight"
add_name = path.split("/")[-1]
folder_list = ["xml","combine_contactlens","original", "images"]
listdir = os.listdir(path)
""
print(listdir)
for name in listdir:
    for folder in folder_list:
        if name == folder:
            path2file = os.path.join(path,folder)
            namein_folder  = os.listdir(path2file)

            for name_file in namein_folder:
                os.rename(os.path.join(path2file,name_file),os.path.join(path2file,add_name+name_file))