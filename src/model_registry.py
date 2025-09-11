from transformers import AutoModelForImageTextToText, Gemma3ForConditionalGeneration

MODEL_REGISTRY = {
    "Qwen2-VL-2B-Instruct": {
        "base_model_path": "/home/pranon/scratch/def-tahmedge/pretrained_models/Qwen2-VL-2B-Instruct",
        "trained_model_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/Qwen2_2bFull",
        "dataset_path": "/home/pranon/projects/def-tahmedge/pranon/llama-factory-training/data/test-cc.json",
        "output_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/Qwen2_2bFull/captions.json",
        "batch_size": 1,
        "generator_class": AutoModelForImageTextToText,
    },
    "Qwen2-VL-7B-Instruct": {
        "base_model_path": "/home/pranon/scratch/def-tahmedge/pretrained_models/Qwen2-VL-7B-Instruct",
        "trained_model_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/Qwen2_7bFull",
        "dataset_path": "/home/pranon/projects/def-tahmedge/pranon/llama-factory-training/data/test-cc.json",
        "output_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/Qwen2_7bFull/captions.json",
        "batch_size": 1,
    },
    "Qwen25-VL-7B-Instruct": {
        "base_model_path": "/home/pranon/scratch/def-tahmedge/pretrained_models/Qwen2.5-VL-7B-Instruct",
        "trained_model_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/Qwen25_7bFull",
        "dataset_path": "/home/pranon/projects/def-tahmedge/pranon/llama-factory-training/data/test-cc.json",
        "output_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/Qwen25_7bFull/captions.json",
        "batch_size": 1,
        "generator_class": AutoModelForImageTextToText,
    },
    "gemma-3-4b-it-1000": {
        "base_model_path": "/home/pranon/scratch/def-tahmedge/pretrained_models/gemma-3-4b-it",
        "trained_model_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/gemma3-3-epoch-1000",
        "dataset_path": "/home/pranon/projects/def-tahmedge/pranon/llama-factory-training/data/test-cc.json",
        "output_path": "/home/pranon/scratch/def-tahmedge/pranon/experiments/gemma3-3-epoch-1000",
        "batch_size": 1,
        "generator_class": Gemma3ForConditionalGeneration,
    },
    "Llama-32-11B-Vision": {"base_model_path": "", "trained_model_path": ""},
}
