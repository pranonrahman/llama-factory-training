from PIL import Image
from torch.utils.data import Dataset


def getimage(image_path):
    with Image.open(image_path) as image:
        return image.convert("RGB")


def getimages(image_paths):
    images = []
    for image_path in image_paths:
        images.append(getimage(image_path))

    return images


class SimplifyMeDataset(Dataset):
    def __init__(self, data_entries):
        self.data_entries = data_entries

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        return self.data_entries[idx]


def get_conversational_formatted_messages(text: str, role='user'):
    if role == 'user':
        return [
            {
                'content': [
                    {
                        'type': 'text',
                        'text': text,
                    },
                    {
                        'type': 'image',
                        'text': None,
                    }
                ],
                'role': role,
            }
        ]
    else:
        return [
            {
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                ],
                "role": role,
            }
        ]
