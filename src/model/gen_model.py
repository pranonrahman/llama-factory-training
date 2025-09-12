import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, GenerationConfig, AutoTokenizer

from src.dataset.collator import collate_fn


@torch.no_grad()
def generate_captions_batch(model, processor, batch_data):
    model.eval()

    try:
        generation_config = GenerationConfig(
            max_new_tokens=256,  # captions don’t need 512
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
        )

        generated_ids = model.generate(
            **batch_data["inputs"],
            generation_config=generation_config,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        input_length = batch_data["inputs"]["input_ids"].shape[-1]
        new_tokens_batch = generated_ids[:, input_length:]

        # Decode responses
        responses = []
        for new_tokens in new_tokens_batch:
            response_text = processor.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
            responses.append(response_text.strip())

        return responses

    except torch.cuda.OutOfMemoryError:
        print("GPU memory exceeded during generation. Try reducing batch size.")
        torch.cuda.empty_cache()
        return None


def generate_output(
        config, dataset, processed_dataset, model, processor, trained_model: bool = False
):
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor, config),
    )

    idx = 0

    for batch in tqdm(dataloader):
        if batch is None:
            continue

        outputs = generate_captions_batch(model, processor, batch)

        if outputs:
            for output in outputs:
                # print(f'\nOriginal chose: {processed_dataset[idx]['chosen']}\nGenerated:{output}')
                processed_dataset[idx][
                    f'{"trained" if trained_model else "base"}_model_output'
                ] = output
                idx += 1

    return processed_dataset


def get_model(config, trained_model: bool = False):
    # Load processor from base model (which should have the chat template)
    processor = AutoProcessor.from_pretrained(
        config["base_model_path"], trust_remote_code=True, use_fast=True
    )

    model = config["generator_class"].from_pretrained(
        config["base_model_path"], trust_remote_code=True
    )

    if trained_model:
        model = PeftModel.from_pretrained(model, config["trained_model_path"])
        model = model.merge_and_unload()

    if not hasattr(processor, 'chat_template') or processor.chat_template is None:
        base_tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])
        if hasattr(base_tokenizer, 'chat_template') and base_tokenizer.chat_template:
            processor.chat_template = base_tokenizer.chat_template
        else:
            processor.chat_template = get_llama_chat_template()

    return processor, model


def get_llama_chat_template():
    """Return the standard Llama chat template"""
    return """
{% if messages[0]['role'] == 'system' %}
{% set loop_messages = messages[1:] %}
{% set system_message = messages[0]['content'] %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = false %}
{% endif %}

{% for message in loop_messages %}
    {% if loop.index0 == 0 and system_message != false %}
        <|start_header_id|>system<|end_header_id|>

{{ system_message }}<|eot_id|>
    {% endif %}
    <|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{% endfor %}
{% if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}"""
