from image_loader import load_images
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import math
import numpy as np

images = load_images(path='tests/')

# get model and processor
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
process = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

images_input = process(images=images,text=['fashion picture'], return_tensors='pt', padding=True) # type: ignore

with torch.inference_mode():
    image_features = model.get_image_features(pixel_values=images_input['pixel_values'])
    text_features = model.get_text_features(input_ids=images_input['input_ids'], attention_mask=images_input['attention_mask'])

    # put it into images dictionary
    image_records = []
    for image, embedding in zip(images, image_features.pooler_output): # type: ignore
        image_records.append(
            {
                "filename": image.filename,
                "image": image,
                "embedding": embedding,
            }
        ) # type: ignore


    # text only
    text_mag = math.sqrt((text_features.pooler_output.flatten()**2).sum()) # type: ignore

    # image magnitude
    for i in image_records:
        image_mag = math.sqrt((i['embedding'] ** 2).sum())

        # cos_similarity
        dot_prod = (i['embedding'] * text_features.pooler_output.flatten()).sum() # type: ignore
        denom = text_mag * image_mag

        cos_similarity_score = dot_prod / denom

        # add the cosine similarity score into the dict
        i['score'] = cos_similarity_score


# sort image_records based on score that we give it
sorted_list = [item for item in sorted(image_records, reverse=True, key=lambda x: x['score'])]
for i in sorted_list:
    print(i['score'], i['filename'], i['embedding'][:5])
