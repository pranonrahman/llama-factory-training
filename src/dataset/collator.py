from src.dataset.simplify_me_dataset import (
    get_images,
    get_conversational_formatted_messages,
)


def collate_fn(batch, processor, config):
    entries = [item for item in batch]
    images = [get_images(item["images"]) for item in batch]

    prompts = [
        processor.apply_chat_template(
            get_conversational_formatted_messages(item["instruction"], "user"),
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
