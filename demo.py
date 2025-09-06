import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import json

class CommandGenerator:
    def __init__(self, model_path):
        """Initialize the command generator with the trained model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.valid_commands = [
            'DIR', 'CD', 'IPCONFIG', 'PING', 'TRACERT', 'NETSTAT', 'TASKLIST',
            'TASKKILL', 'SYSTEMINFO', 'WHOAMI', 'VER', 'HELP', 'CLS', 'COPY',
            'MOVE', 'DEL', 'REN', 'MD', 'RD', 'TYPE', 'FIND', 'FINDSTR', 'SORT',
            'MOUNTVOL', 'IF', 'CIPHER', 'BOOTREC', 'PAUSE', 'WINRS', 'DELTREE',
            'REPLACE', 'NBTSTAT', 'WPEUTIL', 'QAppSrv', 'MOVEUSER', 'EXTRACT',
            'TOUCH', 'RASDIAL', 'ScriptRunner', 'MSG', 'MODE', 'DSQuery', 'BCDBOOT'
        ]
    
    def generate_command(self, user_prompt):
        """Generate a command based on user prompt."""
        # Prepare input
        input_text = f"Command to {user_prompt.lower()} [SEP]"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # Generate output
        output = self.model.generate(
            input_ids,
            max_length=50,
            temperature=0.3,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
        
        # Decode and clean up
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract command part
        if '[SEP]' in generated_text:
            command_part = generated_text.split('[SEP]')[-1].strip()
            first_word = command_part.split(' ')[0].upper()
            
            # Validate command
            if first_word in self.valid_commands:
                parts = command_part.split()
                command = parts[0]
                args = ' '.join(parts[1:4]) if len(parts) > 1 else ''
                return f"{command} {args}".strip()
        
        return None

def load_dataset(file_path):
    """Load the training dataset to show examples."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def main():
    # Initialize command generator
    model_path = "./command_model"
    print(f"Loading model from {model_path}...")
    try:
        cmd_gen = CommandGenerator(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load and display some examples from the dataset
    print("\n=== Example Commands from Training Data ===")
    dataset = load_dataset("train_data.jsonl")
    for i, example in enumerate(dataset[:5]):  # Show first 5 examples
        print(f"\nExample {i+1}:")
        print(f"Prompt: {example['prompt']}")
        print(f"Command: {example['completion']}")
    
    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Type a description of what you want to do (or 'exit' to quit):")
    print("Example: 'list all files in current directory'")
    
    while True:
        user_input = input("\nWhat would you like to do? ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input:
            continue
        
        # Generate command
        command = cmd_gen.generate_command(user_input)
        
        if not command:
            print("\nI couldn't generate a valid command for that request.")
            print("Try something like: 'list files', 'show ip config', 'display running processes'")
            continue
        
        print(f"\nSuggested command: {command}")
        
        # Ask for confirmation
        while True:
            execute = input("Execute this command? (y/n/help): ").strip().lower()
            
            if execute == 'y':
                try:
                    print(f"\nExecuting: {command}")
                    print("-" * 50)
                    result = subprocess.run(
                        command,
                        shell=True,
                        text=True,
                        capture_output=True
                    )
                    
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print("Error details:")
                        print(result.stderr)
                    print("-" * 50)
                    
                except Exception as e:
                    print(f"\n❌ Error executing command: {e}")
                break
                
            elif execute == 'n':
                print("Command not executed.")
                break
                
            elif execute == 'help':
                print("\nAvailable options:")
                print("  y - Execute the command")
                print("  n - Cancel and enter a new command")
                print("  help - Show this help message")
            else:
                print("\nPlease enter 'y' to execute, 'n' to cancel, or 'help' for options.")

if __name__ == "__main__":
    main()
