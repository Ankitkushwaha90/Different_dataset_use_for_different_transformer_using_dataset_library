import pandas as pd
from sklearn.model_selection import train_test_split
import json

def load_data(filepath):
    """Load and preprocess the commands dataset."""
    df = pd.read_csv(filepath)
    
    # Filter only Windows commands (cmd source)
    df = df[df['source'] == 'cmd'].copy()
    
    # Create prompt-completion pairs
    df['prompt'] = df['description'].apply(lambda x: f"Command to {x.lower()}")
    df['completion'] = df['name']
    
    return df[['prompt', 'completion']]

def save_datasets(df, train_path, val_path, test_size=0.2, random_state=42):
    """Split data into training and validation sets and save to JSONL format."""
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Save training data
    with open(train_path, 'w', encoding='utf-8') as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps({
                "prompt": row['prompt'],
                "completion": row['completion']
            }) + '\n')
    
    # Save validation data
    with open(val_path, 'w', encoding='utf-8') as f:
        for _, row in val_df.iterrows():
            f.write(json.dumps({
                "prompt": row['prompt'],
                "completion": row['completion']
            }) + '\n')

if __name__ == "__main__":
    # File paths
    input_file = "all_commands.csv"
    train_file = "train_data.jsonl"
    val_file = "val_data.jsonl"
    
    # Process data
    print("Preparing dataset...")
    df = load_data(input_file)
    print(f"Total command examples: {len(df)}")
    
    # Save datasets
    save_datasets(df, train_file, val_file)
    print(f"Training data saved to {train_file}")
    print(f"Validation data saved to {val_file}")
