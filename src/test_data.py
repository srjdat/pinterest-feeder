from image_loader import load_images
from transformers import CLIPProcessor, CLIPModel


# load images
# get image embeddings
# make a list of dicts with name, embedding, category (maybe)
def test_data(path: str = "tests/"):
    # path to images that we are going to embed
    # by default it's set to test/
    image_list = load_images(path=path)
    data = []

    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    process = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    for image in image_list:
        image_info = process(images=image, return_tensors='pt') # type: ignore
        image_features = model.get_image_features(pixel_values=image_info['pixel_values'])

        data.append(
            {
                "image_filename": image.filename,
                "image_embedding": image_features.pooler_output, # type: ignore
                "image": image
            }
        )

    # return the data that we're making
    return data

def main():
    test_data()

if __name__ == "__main__":
    main()
