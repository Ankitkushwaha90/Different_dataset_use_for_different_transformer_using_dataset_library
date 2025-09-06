import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    """Load the fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def generate_command(prompt, model, tokenizer, max_length=50, temperature=0.7):
    """Generate a command based on the prompt with improved output handling."""
    # Prepare the input with clear instruction
    input_text = f"Command to {prompt.lower()}. [SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate output with more focused parameters
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.3,  # Lower temperature for more focused output
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=20,  # Lower top_k for less random sampling
        top_p=0.9,  # Slightly lower top_p for more focused output
        repetition_penalty=1.2,  # Discourage repetition
        no_repeat_ngram_size=2,  # Prevent 2-gram repetition
    )
    
    # Decode and clean up the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the command part (after [SEP])
    if '[SEP]' in generated_text:
        generated_text = generated_text.split('[SEP]')[-1].strip()
    
    # Clean up the command
    import re
    # Remove any non-command text (after first word that's not a command)
    first_word = generated_text.split(' ')[0].upper()
    valid_commands = ['DIR', 'CD', 'IPCONFIG', 'PING', 'TRACERT', 'NETSTAT', 'TASKLIST', 
                     'TASKKILL', 'SYSTEMINFO', 'WHOAMI', 'VER', 'HELP', 'CLS', 'COPY',
                     'MOVE', 'DEL', 'REN', 'MD', 'RD', 'TYPE', 'FIND', 'FINDSTR', 'SORT']
    
    if first_word in valid_commands:
        # If it starts with a valid command, take just that command and its arguments
        parts = generated_text.split()
        command = parts[0]
        # Take up to 3 arguments to prevent command injection
        args = ' '.join(parts[1:4]) if len(parts) > 1 else ''
        generated_text = f"{command} {args}".strip()
    else:
        # If it doesn't start with a valid command, return empty
        generated_text = ''
    
    return generated_text

def main():
    # Load the fine-tuned model
    model_path = "./command_model"
    print(f"Loading model from {model_path}...")
    try:
        model, tokenizer = load_model(model_path)
        print("✅ Model loaded successfully!")
        print("\nType a description of what you want to do, for example:")
        print("  - list files in current directory")
        print("  - show network configuration")
        print("  - display system information")
        print("  - show running processes")
        print("\nType 'exit' to quit.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you've trained the model first by running 'python train.py'")
        return
    
    # Interactive loop
    while True:
        user_input = input("\nWhat would you like to do? ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input:
            continue
            
        # Generate command
        try:
            command = generate_command(user_input, model, tokenizer)
            
            if not command:
                print("\nI couldn't generate a valid command for that request. Please try rephrasing.")
                print("For example, try: 'list files', 'show ip config', 'display running processes'")
                continue
                
            print(f"\nSuggested command: {command}")
            
            # Show a warning for potentially dangerous commands
            dangerous_commands = ['DEL', 'RMDIR', 'RD', 'FORMAT', 'CHKDSK', 'DISKPART']
            if any(cmd in command.upper().split() for cmd in dangerous_commands):
                print("\n⚠️  WARNING: This command could be dangerous. Make sure you understand what it does before executing.")
            
            # Ask if user wants to execute the command
            while True:
                execute = input("\nExecute this command? (y/n/help): ").strip().lower()
                if execute == 'y':
                    import subprocess
                    try:
                        print(f"\nExecuting: {command}")
                        print("-" * 50)
                        result = subprocess.run(
                            command, 
                            shell=True, 
                            check=True,
                            text=True,
                            capture_output=True
                        )
                        if result.stdout:
                            print(result.stdout)
                        if result.stderr:
                            print("Error details:")
                            print(result.stderr)
                        print("-" * 50)
                    except subprocess.CalledProcessError as e:
                        print("\n❌ Error executing command:")
                        print(f"Command: {e.cmd}")
                        print(f"Return code: {e.returncode}")
                        if e.stderr:
                            print("\nError details:")
                            print(e.stderr)
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
        except Exception as e:
            print(f"Error generating command: {e}")

if __name__ == "__main__":
    main()
