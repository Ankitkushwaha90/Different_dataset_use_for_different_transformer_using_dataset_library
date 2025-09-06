import json
import random
from typing import List, Dict, Tuple

# Base command templates with variations
COMMAND_TEMPLATES = {
    'DIR': [
        'list files in {dir}',
        'show contents of {dir}',
        'what files are in {dir}',
        'display directory {dir}',
        'show me files in {dir}'
    ],
    'IPCONFIG': [
        'show network configuration',
        'display ip address',
        'what is my ip',
        'show network settings',
        'display network information'
    ],
    'TASKLIST': [
        'show running processes',
        'list all running programs',
        'what processes are running',
        'display active tasks',
        'show me running applications'
    ],
    'SYSTEMINFO': [
        'show system information',
        'display computer specs',
        'what are my system details',
        'show hardware information',
        'display system configuration'
    ],
    'PING': [
        'ping {host}',
        'test connection to {host}',
        'check if {host} is online',
        'test network connectivity to {host}',
        'is {host} reachable'
    ],
    'NETSTAT': [
        'show network connections',
        'display active connections',
        'what ports are open',
        'show network statistics',
        'display listening ports'
    ],
    'WHOAMI': [
        'who am i',
        'show current user',
        'what is my username',
        'display current user information',
        'which user is logged in'
    ]
}

# Common directories and hosts for placeholders
DIRECTORIES = ['current directory', 'this folder', 'desktop', 'documents', 'downloads', 'C:\\']
HOSTS = ['google.com', '8.8.8.8', 'localhost', 'microsoft.com', 'github.com']

def generate_prompt_variations(base_prompt: str) -> List[str]:
    """Generate multiple variations of a base prompt."""
    variations = [base_prompt]
    
    # Add some common variations
    if '{dir}' in base_prompt:
        for directory in DIRECTORIES:
            variations.append(base_prompt.format(dir=directory))
    elif '{host}' in base_prompt:
        for host in HOSTS:
            variations.append(base_prompt.format(host=host))
    
    # Add some random variations
    for _ in range(2):
        if random.random() > 0.5:
            variations.append(f"how to {base_prompt}")
        if random.random() > 0.5:
            variations.append(f"command to {base_prompt}")
        if random.random() > 0.5:
            variations.append(f"what is the command for {base_prompt}")
    
    return variations

def create_dataset_entry(command: str, description: str) -> Dict:
    """Create a dataset entry with the given command and description."""
    return {
        "prompt": description,
        "completion": command
    }

def generate_dataset() -> List[Dict]:
    """Generate a dataset of command examples."""
    dataset = []
    
    # Generate entries for each command template
    for cmd, templates in COMMAND_TEMPLATES.items():
        for template in templates:
            variations = generate_prompt_variations(template)
            for var in variations:
                dataset.append(create_dataset_entry(cmd, var))
    
    # Add some common command variations
    common_commands = [
        ("CLS", "clear the screen"),
        ("HELP", "show help"),
        ("VER", "show windows version"),
        ("TIME", "display current time"),
        ("DATE", "show current date"),
        ("ECHO Hello World", "print hello world"),
        ("TASKKILL /F /IM notepad.exe", "close notepad"),
        ("START notepad", "open notepad"),
        ("MKDIR new_folder", "create a new folder"),
        ("RMDIR /S /Q folder", "delete a folder and all contents"),
        ("COPY file.txt newfile.txt", "copy a file"),
        ("MOVE file.txt C:\\temp\\", "move a file"),
        ("TYPE file.txt", "display file contents"),
        ("FIND \"search\" file.txt", "find text in a file"),
        ("SORT file.txt", "sort lines in a file"),
    ]
    
    for cmd, desc in common_commands:
        dataset.append(create_dataset_entry(cmd, desc))
        # Add variations
        dataset.append(create_dataset_entry(cmd, f"how to {desc}"))
        dataset.append(create_dataset_entry(cmd, f"command to {desc}"))
    
    return dataset

def save_dataset(dataset: List[Dict], filename: str):
    """Save the dataset to a JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')

def main():
    # Generate the dataset
    print("Generating dataset...")
    dataset = generate_dataset()
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Split into train and validation sets (80/20 split)
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    # Save the datasets
    save_dataset(train_data, "train_cmd_data.jsonl")
    save_dataset(val_data, "val_cmd_data.jsonl")
    
    print(f"Dataset created successfully!")
    print(f"Total examples: {len(dataset)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print("\nSample training examples:")
    for i, example in enumerate(train_data[:5]):
        print(f"{i+1}. Prompt: {example['prompt']}")
        print(f"   Command: {example['completion']}\n")

if __name__ == "__main__":
    main()
