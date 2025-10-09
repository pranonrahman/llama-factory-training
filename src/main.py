import argparse
import gc
import json

import torch

from src.dataset.simplify_me_dataset import SimplifyMeDataset
from src.model.gen_model import get_model, generate_output
from src.model_registry import MODEL_REGISTRY


def main(config, few_shot=False):
    with open(config['dataset_path'], 'r', encoding='utf-8') as f:
        processed_dataset = json.load(f)
        dataset = SimplifyMeDataset(processed_dataset)

    if not config["trained"]:
        processor, model = get_model(config)
        model = model.to('cuda').eval()

        processed_dataset = generate_output(config, dataset, processed_dataset, model, processor, False, few_shot)
    else:
        processor, model = get_model(config, True)
        model = model.to('cuda').eval()

        with open(config['dataset_path'], 'r', encoding='utf-8') as f:
            processed_dataset = json.load(f)
            dataset = SimplifyMeDataset(processed_dataset)

        processed_dataset = generate_output(config, dataset, processed_dataset, model, processor, True, few_shot)

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    with open(config['output_path'], 'w', encoding='utf-8') as f:
        json.dump(processed_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run model inference on dataset")
        parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help=f"Name of the model to use. Available: {list(MODEL_REGISTRY.keys())}"
        )
        parser.add_argument(
            "--few_shot",
            action="store_true",
            required=False,
            default=False,
            help=f"Whether to use few-shot or not. Default: False"
        )
        args = parser.parse_args()

        if args.model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{args.model_name}' not found in MODEL_REGISTRY. "
                             f"Available: {list(MODEL_REGISTRY.keys())}")

        main(MODEL_REGISTRY[args.model_name], args.few_shot)
