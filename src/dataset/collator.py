from src.dataset.simplify_me_dataset import getimages, get_conversational_formatted_messages


def collate_fn(batch, processor, config):
    entries = [item for item in batch]
    images = [getimages(item['images']) for item in batch]

    if config['base_model_path'] != '/home/pranon/scratch/def-tahmedge/pretrained_models/Llama-3.2-11B-Vision':
        prompts = [get_conversational_formatted_messages(item["instruction"], "user")
                   for item in batch]
    else:
        prompts = [
            f"<|image|><|begin_of_text|>{item["instruction"]}" for item in batch
        ]

    ids = [item['id'] for item in batch]

    # Process batch
    try:
        inputs = processor(text=prompts, images=images, return_tensors='pt', padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        return {
            'entries': entries,
            'inputs': inputs,
            'ids': ids
        }
    except Exception as e:
        # Fallback to smaller batch if memory issues
        print(f"Batch processing failed: {e}. Falling back to individual processing.")
        return None
