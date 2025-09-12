from transformers import (
    AutoModelForImageTextToText,
    Gemma3ForConditionalGeneration,
    MllamaForConditionalGeneration,
)

MODEL_REGISTRY = {
    "gemma-3-4b-it-base": {
        "base_model_path": "/home/pranon/scratch/def-tahmedge/pretrained_models/gemma-3-4b-it",
        "trained_model_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/gemma3-3-epoch-1000",
        "dataset_path": "/home/pranon/projects/def-tahmedge/pranon/llama-factory-training/data/test-cc.json",
        "output_path": " /home/pranon/scratch/def-tahmedge/pranon/experiments/gen-captions/gemma3-base.json",
        "batch_size": 48,
        "generator_class": Gemma3ForConditionalGeneration,
        "trained": False,
    },
    "gemma-3-4b-it-1000-trained": {
        "base_model_path": "/home/pranon/scratch/def-tahmedge/pretrained_models/gemma-3-4b-it",
        "trained_model_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/gemma3-3-epoch-1000",
        "dataset_path": "/home/pranon/projects/def-tahmedge/pranon/llama-factory-training/data/test-cc.json",
        "output_path": " /home/pranon/scratch/def-tahmedge/pranon/experiments/gen-captions/gemma3-3-epoch-1000-trained.json",
        "batch_size": 48,
        "generator_class": Gemma3ForConditionalGeneration,
        "trained": True,
    },
    "gemma-3-4b-it-10000-trained": {
        "base_model_path": "/home/pranon/scratch/def-tahmedge/pretrained_models/gemma-3-4b-it",
        "trained_model_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/gemma3-3-epoch-10000",
        "dataset_path": "/home/pranon/projects/def-tahmedge/pranon/llama-factory-training/data/test-cc.json",
        "output_path": " /home/pranon/scratch/def-tahmedge/pranon/experiments/gen-captions/gemma3-3-epoch-10000-trained.json",
        "batch_size": 48,
        "generator_class": Gemma3ForConditionalGeneration,
        "trained": True,
    },
    "gemma-3-4b-it-full-trained": {
        "base_model_path": "/home/pranon/scratch/def-tahmedge/pretrained_models/gemma-3-4b-it",
        "trained_model_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/gemma3-1-epoch-full-ds",
        "dataset_path": "/home/pranon/projects/def-tahmedge/pranon/llama-factory-training/data/test-cc.json",
        "output_path": " /home/pranon/scratch/def-tahmedge/pranon/experiments/gen-captions/gemma3-3-epoch-full-trained.json",
        "batch_size": 48,
        "generator_class": Gemma3ForConditionalGeneration,
        "trained": True,
    },
    "Llama-32-11B-Vision": {
        "base_model_path": "/home/pranon/scratch/def-tahmedge/pretrained_models/Llama-3.2-11B-Vision",
        "trained_model_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/llama32-3-epoch-1000/",
        "dataset_path": "/home/pranon/projects/def-tahmedge/pranon/llama-factory-training/data/test-cc.json",
        "output_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/llama32-3-epoch-1000/captions.json",
        "batch_size": 48,
        "generator_class": MllamaForConditionalGeneration,
    },
}
