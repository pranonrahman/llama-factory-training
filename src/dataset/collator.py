from src.dataset.simplify_me_dataset import (
    get_images,
    get_conversational_formatted_messages, get_few_shot_conversational_formatted_messages,
)


def collate_fn(batch, processor, config, few_shot):
    few_shot_images = [
        "/home/pranon/scratch/def-tahmedge/simplify-me-dataset/simplify-me/compressed_images/flickr30k_val/3025817244.jpg",
        "/home/pranon/scratch/def-tahmedge/simplify-me-dataset/simplify-me/compressed_images/flickr30k_val/213609234.jpg",
        "/home/pranon/scratch/def-tahmedge/simplify-me-dataset/simplify-me/compressed_images/coco_2014_val/COCO_val2014_000000015260.jpg",
        "/home/pranon/scratch/def-tahmedge/simplify-me-dataset/simplify-me/compressed_images/viswiz_val/VizWiz_val_00004571.jpg",
        "/home/pranon/scratch/def-tahmedge/simplify-me-dataset/simplify-me/compressed_images/textcaps_train/d49125792e6caace.jpg"
    ]

    entries = [item for item in batch]

    if not few_shot:
        images = [get_images(item["images"]) for item in batch]
    else:
        images = [get_images(few_shot_images + item["images"]) for item in batch]

    if not few_shot:
        prompts = [
            processor.apply_chat_template(
                get_conversational_formatted_messages(item["instruction"], "user"),
                add_generation_prompt=True,
            )
            for item in batch
        ]
    else:
        prompts = [
            processor.apply_chat_template(
                get_few_shot_conversational_formatted_messages(item["instruction"], "user"),
                add_generation_prompt=True,
            )
            for item in batch
        ]

    ids = [item["id"] for item in batch]

    # Process batch
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        # max_length=2048,
        # truncation=True,
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    return {"entries": entries, "inputs": inputs, "ids": ids}
