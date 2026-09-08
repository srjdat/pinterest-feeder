from PIL import Image
import os
from dotenv import load_dotenv

"""
    Loads images from given file path in the .env file. 
    Checks if path exists, if so go through and get all the images. 
"""
def load_images(): 
    
    images = [] # where all the images will go
    VALID_FORMATS = ('.png', '.jpg', '.jpeg', '.webp')

    # check if you can find images folder
    load_dotenv()
    path = os.getenv('FILE_PATH') # open all images in this folder 
    num = 0 # naming scheme
    if path: 
        for i in os.scandir(path=path): 
            if i.name.lower().endswith(VALID_FORMATS): # make sure it's actually an image
                img = Image.open(i)
                images.append(img) 

    print(len(images))

    return images


def main(): 
    load_images()


if __name__ == "__main__":
    main()

