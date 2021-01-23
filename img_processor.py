from PIL import Image
import os, sys

paths = [
    "./20_80/",
    "./50_50/",
    "./test_set/"
    ]

def resize():
    for path in paths:
        print("[INFO] Resising images in " + path)
        dirs = os.listdir(path)
        for item in dirs:
            if os.path.isfile(path+item):
                im = Image.open(path+item).convert('L')
                f, e = os.path.splitext(path+item)
                imResize = im.resize((64,64), Image.ANTIALIAS)
                imResize.save(f + '.jpg', 'JPEG', quality=100)
        print("Done.")

resize()
