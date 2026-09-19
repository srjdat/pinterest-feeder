import json
from typing import Any
from pathlib import Path

from transformers.image_utils import ImageFeatureExtractionMixin
from image_loader import load_images
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
import sys
from memory_profiler import profile
import resource

dataset_list = [] # type: ignore

def create_dataset(image_list: list[Any]):
    global dataset_list
    dataset_list = [] # make it empty every time we call it again to create a new dadtaset

    # loop through all the images
    for image in image_list:
        # see if name contains fashion, anime, or silver
        # based on if it contains those create category: fashion, car, anime
        if "fashion" in image.filename:
            dataset_list.append(
                {
                    "category": "fashion",
                    "image_filename": image.filename,
                    "queries": []
                }
            )
        elif "silver" in image.filename:
            dataset_list.append(
                {
                    "category": "car",
                    "image_filename": image.filename,
                    "queries": []
                }
            )
        elif "anime" in image.filename:
            dataset_list.append(
                {
                    "category": "anime",
                    "image_filename": image.filename,
                    "queries": []
                }
            )

    # dump this dataset into the json file
    with open('data.json', 'w') as outfile:
        json.dump(dataset_list, outfile, indent=2)

# @profile
def dataset(**kwargs):

    # create the dataset without any labels right now
    path = kwargs.get('path')
    if path != None:
        images_list = load_images(path=path)
    else:
        images_list = load_images()

    # i'm creating a variable so i can change it later if need me also just cleaner
    json_file = Path('data.json')
    json_created = True if json_file.is_file() else False

    # read json file
    if json_created: # if we already created the json file and we have update labels
        with open('data.json', 'r') as infile:
            data_list = json.load(infile)
    else: # if we haven't created the json file
        create_dataset(images_list)
        print("please run this file again, there was no dataset to read from")
        sys.exit()

    # create a list that we are going to return
    return_list = []

    # create the model and processor so we can get the image and query embeddings to combine
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    process = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    for image, item in zip(images_list, data_list):
        # create the image embedding to use
        image_info = process(images=image, return_tensors='pt') # type: ignore
        image_features = model.get_image_features(pixel_values=image_info['pixel_values'])

        # item is a dict
        for query, label in zip(item['queries'], item['label']):
            text_info= process(text=query, return_tensors='pt', padding=True) # type: ignore
            text_features = model.get_text_features(input_ids=text_info['input_ids'], attention_mask=text_info['attention_mask'])

            # [1, 1024] tensor size
            concatenated_embedding = torch.cat((image_features.pooler_output, text_features.pooler_output), 1) #type: ignore

            # append to the list that we're going to return
            return_list.append(
                {
                    "image_filename": item['image_filename'],
                    "query": query,
                    "label": label,
                    "input": concatenated_embedding
                }
            )

    return return_list
def main():
    dataset()


if __name__ == "__main__":
    # usage = resource.getrusage(resource.RUSAGE_SELF)
    # print(usage.ru_maxrss )
    main()
