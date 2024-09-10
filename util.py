import os

path = "/home/yyh/lab/vggsfm/data/car/images"

# read image file names, if the name is 6 digits, add prefix "frame_"

images = os.listdir(path)
for image in images:
    if len(image) == 10:
        os.rename(path + "/" + image, path + "/" + "frame_" + image)