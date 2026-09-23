import torch
from model import PinterestFeeder
from test_data import test_data
from transformers import CLIPProcessor, CLIPModel

model = PinterestFeeder(1024)
model.load_state_dict(torch.load("models/pinterest_feeder1.pth", weights_only=True))
model.eval()

# path is automatically set to test folder so no need right here but you can change it if test folder is somewhere else
data = test_data()
queries = ['anime art', 'silver luxury car', 'fashion inspiration']
# print(data)

unique_images = list(set(item['image_filename'] for item in data))

validation_data = [
    item for item in data
    if item['image_filename'] in unique_images
]

embed_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
process = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

embedding_list = []

for image in data:
    image_info = process(images=image['image'], return_tensors='pt') # type: ignore
    image_features = embed_model.get_image_features(pixel_values=image_info['pixel_values'])

    for query in queries:
        text_info= process(text=query, return_tensors='pt', padding=True) # type: ignore
        text_features = embed_model.get_text_features(input_ids=text_info['input_ids'], attention_mask=text_info['attention_mask'])

        embedding_list.append(
            {
                "embedding": torch.cat((image_features.pooler_output, text_features.pooler_output), 1), # type: ignore
                "query": query
            }
        )

with torch.inference_mode():
    for embedding, image in zip(embedding_list, data):
        score = model(embedding['embedding'][0])
        print(f"score {torch.sigmoid(score)}, image {image['image_filename']}, query {embedding['query']}")
