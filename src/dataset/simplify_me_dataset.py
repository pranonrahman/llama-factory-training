from PIL import Image
from torch.utils.data import Dataset


def get_image(image_path):
    with Image.open(image_path) as image:
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        return image.convert("RGB")


def get_images(image_paths):
    images = []
    for image_path in image_paths:
        images.append(get_image(image_path))

    return images


class SimplifyMeDataset(Dataset):
    def __init__(self, data_entries):
        self.data_entries = data_entries

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        return self.data_entries[idx]


def get_few_shot_conversational_formatted_messages(text: str, role="user"):
    if role == "user":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in simple terms."},
                    {"type": "image", "image": "image1.jpg"}
                ]
            },
            {
                "role": "assistant",
                "content": "A girl bounces on a large red ball."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in simple terms."},
                    {"type": "image", "image": "image2.jpg"}
                ]
            },
            {
                "role": "assistant",
                "content": "One man and two women wearing shorts, walk down the street."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in simple terms."},
                    {"type": "image", "image": "image3.jpg"}
                ]
            },
            {
                "role": "assistant",
                "content": "A cake cover is made to look like a wire birdcage. "
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in simple terms."},
                    {"type": "image", "image": "image4.jpg"}
                ]
            },
            {
                "role": "assistant",
                "content": "A person is holding a blue Wii remote in a living room."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in simple terms."},
                    {"type": "image", "image": "image5.jpg"}
                ]
            },
            {
                "role": "assistant",
                "content": "A bottle of double wide India pale ale.sits in front of a window."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in simple terms."},
                    {"type": "image", "image": "new_image.jpg"}
                ]
            }
        ]
    else:
        return [
            {
                "content": [
                    {"type": "text", "text": text},
                ],
                "role": role,
            }
        ]


def get_conversational_formatted_messages(text: str, role="user"):
    if role == "user":
        return [
            {
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    },
                    {
                        "type": "image",
                        "text": None,
                    },
                ],
                "role": role,
            }
        ]
    else:
        return [
            {
                "content": [
                    {"type": "text", "text": text},
                ],
                "role": role,
            }
        ]
