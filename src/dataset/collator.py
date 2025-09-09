from src.dataset.simplify_me_dataset import getimages, get_conversational_formatted_messages


def collate_fn(batch, processor):
    entries = [item for item in batch]
    images = [getimages(item['images']) for item in batch]
    prompts = [
        processor.apply_chat_template(
            get_conversational_formatted_messages(item["instruction"], "user"), add_generation_prompt=True) for item in batch]
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
