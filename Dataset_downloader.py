from huggingface_hub import snapshot_download
import os

# Create the destination folder
save_dir = "pocketdoc_datasets"
os.makedirs(save_dir, exist_ok=True)

datasets = [
        "PocketDoc/Dans-MemoryCore-CoreCurriculum-Small",
        "PocketDoc/Dans-Mathmaxx",
        "PocketDoc/Dans-Mathmaxx-Numina-CoT",
        "PJMixers/Math-Multiturn-1K-ShareGPT",
        "PocketDoc/Dans-Benchmaxx-COT",
        "PocketDoc/Dans-Codemaxx-LeetCode",
        "PocketDoc/Dans-Codemaxx-CodeFeedback-Conversations",
        "PocketDoc/Dans-Codemaxx-Bigcode-SelfInstruct",
        "PocketDoc/Dans-Taskmaxx",
        "PocketDoc/Dans-Taskmaxx-DataPrepper",
        "PocketDoc/Dans-Taskmaxx-ConcurrentQA-Reworked",
        "PocketDoc/Dans-Taskmaxx-TableGPT",
        "PocketDoc/Dans-Taskmaxx-SciRIFF",
        "PocketDoc/Dans-Taskmaxx-Edit",
        "PocketDoc/Dans-Toolmaxx-Agent",
        "PocketDoc/Dans-Toolmaxx-ShellCommands",
        "PocketDoc/Dans-Toolmaxx-Functions-Toolbench",
        "PocketDoc/Dans-Toolmaxx-Functions-ToolACE",
        "PocketDoc/Dans-Assistantmaxx-Sharegpt",
        "PocketDoc/Dans-Assistantmaxx-OpenAssistant2",
        "PocketDoc/Dans-Assistantmaxx-Opus-Merge",
        "PocketDoc/Dans-Assistantmaxx-Synthia",
        "PocketDoc/Dans-Assistantmaxx-ASL",
        "PocketDoc/Dans-Assistantmaxx-PersonaLLM-Opus",
        "PocketDoc/Dans-Assistantmaxx-UnnaturalInstructions-GPT4",
        "PocketDoc/Dans-Assistantmaxx-LongAlign",
        "PocketDoc/Dans-Assistantmaxx-Camel-GPT4",
        "PocketDoc/Dans-Assistantmaxx-OpenLeecher-Instruct",
        "PocketDoc/Dans-Systemmaxx",
        "PocketDoc/Dans-Logicmaxx-Skunkworks",
        "PocketDoc/Dans-Logicmaxx-FI-VeriMed",
        "PocketDoc/Dans-Logicmaxx-SAT-AP",
        "PocketDoc/Dans-Logicmaxx-Magpie-Ultra",
        "PJMixers/grimulkan_theory-of-mind-ShareGPT",
        "PJMixers/grimulkan_physical-reasoning-ShareGPT",
        "NewEden/Claude-Instruct-5K",
        "NewEden/Hydrus-HelpSteer2",
        "NewEden/Hydrus-Instruct-SmolTalk",
        "NewEden/Hydrus-SonnetOrca",
        "NewEden/Science-QA-sharegpt",
        "NewEden/GSM8K-R1-filtered",
        "NewEden/No_Robots-R1-Filtered",
        "NewEden/Tulu-Personas-Filtered-Sharegpt",
        "NewEden/Hydrus-Chat_error-Pure-Dove-sharegpt",
        "NewEden/vanilla-backrooms-claude-sharegpt",
        "NewEden/Hydrus_Anthropic_hh_harmful-sharegpt",
        "NewEden/Kalo-Opus-Instruct-22k-Refusal-Murdered",
        "NewEden/Claude-Instruct-2.7K",
        "NewEden/Claude-Instruct-2.7K"
    ]

for dataset in datasets:
    dataset_dir = os.path.join(save_dir, dataset.replace("/", "_"))
    os.makedirs(dataset_dir, exist_ok=True)
    
    snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        local_dir=dataset_dir,
        ignore_patterns=["*.pt", "*.bin", "*.ckpt"],  # Skip large binary files if any
    )
    print(f"Downloaded {dataset} to {dataset_dir}")