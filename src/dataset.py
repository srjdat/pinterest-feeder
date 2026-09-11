from image_loader import load_images
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

def dataset(**kwargs): 
    queries_list = ['anime picture', 'silver supercar picture', 'fashion picture']
    path = kwargs.get('path')
    if path != None:
        images_list = load_images(path=path)
    else: 
        images_list = load_images()

    dataset = []
    # manually create dataset
    for query in queries_list: 
        for images in images_list:
            # label_input = input(f"query: {query} image: {images.filename} > ")

            if query.split()[0] in images.filename: 
                label_input = 1.0
            else: 
                label_input = 0.0

            dataset.append(
                {
                    "query": query, 
                    "image_filename": images.filename,
                    "label": label_input,
                }
            )

    # create embeddings
    # get model and processor 
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    process = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    processed_info = process(images=images_list, text=queries_list, return_tensors='pt', padding=True) # type: ignore

    # get the embeddings (tensor values)
    image_embeddings = {}
    query_embeddings = {}

    with torch.inference_mode():
        image_features = model.get_image_features(pixel_values=processed_info['pixel_values']) 
        text_features = model.get_text_features(input_ids=processed_info['input_ids'], attention_mask=processed_info['attention_mask'])

        for image, embedding in zip(images_list, image_features.pooler_output): #type: ignore
            image_embeddings.update(
                {
                    f"{image.filename}": embedding
                }
            )
        for text, embedding in zip(queries_list, text_features.pooler_output): # type: ignore
            query_embeddings.update(
                {
                    f"{text}": embedding
                }
            )

    inputs = []
    labels = []
    for item in dataset: 
        q_emb = query_embeddings[item['query']]
        i_emb = image_embeddings[item['image_filename']]

        # we get the combined embeddings creating a 1024 size tensor array that we use for our neural network
        combined = torch.cat((q_emb, i_emb), 0)
        item.update( # add this into our dataset list with each dictionary being an image
            {
                "input": combined
            }
        )

    return (dataset)

def main(): 
    dataset()

if __name__ == "__main__":
    main()
