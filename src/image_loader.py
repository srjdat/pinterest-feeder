from typing import Any
from PIL import Image
import os
from dotenv import load_dotenv

"""
    Loads images from given file path in the .env file.
    Checks if path exists, if so go through and get all the images.
"""
def load_images(**kwargs)-> list[Any]: # if they give us a path otherwise use the default one

    images = [] # where all the images will go
    VALID_FORMATS = ('.png', '.jpg', '.jpeg', '.webp')

    # check if you can find images folder
    load_dotenv()
    path = kwargs.get('path', os.getenv('FILE_PATH')) # open all images in this folder
    print("Hello")
    if path:
        for i in os.scandir(path=path):
            if i.name.lower().endswith(VALID_FORMATS): # make sure it's actually an image
                img = Image.open(i)
                images.append(img)

    return images


def main():
    load_images()


if __name__ == "__main__":
    main()
