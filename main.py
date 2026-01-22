import argparse
import ollama
import numpy as np
import pandas as pd
import os

def run_prompt_on_image(image_path: str, prompt='', model='ministral-3:3b') -> str:
    """
    Runs a text prompt on an image using Ollama's vision model and returns the response text.
    
    Args:
        image_path (str): Path to the image file (e.g., 'path/to/image.jpg').
        prompt (str): The text prompt to run on the image.
    
    Returns:
        str: The generated text response from the model.
    
    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: For other Ollama-related errors (e.g., model not available).
    """
    try:
        if prompt is None or prompt.strip() == "":
            prompt = "Describe the image briefly."
        
        # Send the prompt and image to Ollama
        response = ollama.chat(
            model,  # Vision-capable model; ensure it's pulled in Ollama
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]  # Ollama accepts file paths directly
                }
            ]
        )
        return response['message']['content']
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}. Ensure Ollama app is running and the model is available.")

def compute_embedding(text: str, model: str = 'embeddinggemma') -> np.ndarray:
    """
    Computes the embedding of a given text string using Ollama's embedding model.
    
    Args:
        text (str): The input text to embed.
        model (str): The Ollama model to use (default: 'embeddinggemma').
    
    Returns:
        np.ndarray: The embedding as a NumPy array of floats.
    
    Raises:
        Exception: For Ollama-related errors (e.g., model not available).
    """
    try:
        response = ollama.embeddings(model=model, prompt=text)
        embedding = response['embedding']  # List of floats
        return np.array(embedding)
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}. Ensure Ollama app is running and the model is available.")

def save_model(model_df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the model DataFrame to a CSV file, serializing the 'embedding' column as strings.
    
    Args:
        model_df (pd.DataFrame): The model DataFrame to save.
        filepath (str): Path to the output CSV file (e.g., 'model.csv').
    """
    df_copy = model_df.copy()
    # Serialize embeddings to comma-separated strings
    df_copy['embedding'] = df_copy['embedding'].apply(lambda x: ','.join(map(str, x)))
    df_copy.to_csv(filepath, index=False)
    print(f"Model saved to {filepath}")

def load_model(filepath: str) -> pd.DataFrame:
    """
    Loads the model DataFrame from a CSV file, deserializing the 'embedding' column back to NumPy arrays.
    
    Args:
        filepath (str): Path to the input CSV file (e.g., 'model.csv').
    
    Returns:
        pd.DataFrame: The loaded model DataFrame.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    df = pd.read_csv(filepath)
    # Deserialize embeddings back to NumPy arrays
    df['embedding'] = df['embedding'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
    print(f"Model loaded from {filepath}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Process images and update the model CSV.")
    parser.add_argument('text_file', help='Path to text file with image filenames (one per line)')
    parser.add_argument('model_file', help='Path to the model CSV file')
    args = parser.parse_args()
    
    # Read image filenames from text file
    with open(args.text_file, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]
    
    image_dir = 'frames'  # Assumes images are in 'frames/' directory
    data = []
    
    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping.")
            continue
        print(f"Processing {filename}...")
        description = run_prompt_on_image(image_path, 'Describe what you see in this image, briefly.')
        embedding = compute_embedding(description)
        data.append({
            'filename': filename,
            'description': description,
            'embedding': embedding
        })
    
    # Load existing model
    model_df = load_model(args.model_file)
    
    # Create DataFrame from new data
    new_df = pd.DataFrame(data)
    
    # Update model by concatenating
    updated_df = pd.concat([model_df, new_df], ignore_index=True)
    
    # Save updated model
    save_model(updated_df, args.model_file)
    print("Model updated and saved.")

if __name__ == "__main__":
    main()