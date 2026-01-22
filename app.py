from flask import Flask, request, render_template_string, send_from_directory
import ollama
import numpy as np
import pandas as pd
import os

# Include the necessary functions (copied from your notebook/script for completeness)
def compute_embedding(text: str, model: str = 'embeddinggemma') -> np.ndarray:
    try:
        response = ollama.embeddings(model=model, prompt=text)
        embedding = response['embedding']
        return np.array(embedding)
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}")

def load_model(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['embedding'] = df['embedding'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
    return df

def find_most_similar(model_df: pd.DataFrame, new_embedding: np.ndarray) -> tuple[str, str]:
    if model_df.empty:
        raise ValueError("Model DataFrame is empty.")
    
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    similarities = [cosine_similarity(new_embedding, emb) for emb in model_df['embedding']]
    max_idx = np.argmax(similarities)
    return model_df.iloc[max_idx]['filename'], model_df.iloc[max_idx]['description']

# Initialize Flask app and load model
app = Flask(__name__)
MODEL_FILE = 'model.csv'  # Path to your model CSV
try:
    model_df = load_model(MODEL_FILE)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: {MODEL_FILE} not found. Please ensure the model CSV exists.")
    model_df = pd.DataFrame()  # Empty fallback

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            return render_template_string('''
            <h1>Error</h1>
            <p>Please enter a query.</p>
            <a href="/">Back</a>
            ''')
        
        try:
            embedding = compute_embedding(query)
            filename, desc = find_most_similar(model_df, embedding)
            return render_template_string('''
            <h1>Search Result</h1>
            <p><strong>Filename:</strong> {{ filename }}</p>
            <p><strong>Description:</strong> {{ desc }}</p>
            <img src="/images/{{ filename }}" alt="Matching Image" style="max-width: 500px;">
            <br><br>
            <a href="/">Search Again</a>
            ''', filename=filename, desc=desc)
        except Exception as e:
            return render_template_string('''
            <h1>Error</h1>
            <p>{{ error }}</p>
            <a href="/">Back</a>
            ''', error=str(e))
    
    return render_template_string('''
    <h1>Image Search</h1>
    <form method="post">
        <label for="query">Enter your search query:</label><br>
        <input type="text" id="query" name="query" placeholder="e.g., A bear in the forest" required>
        <br><br>
        <button type="submit">Search</button>
    </form>
    ''')

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from the frames directory."""
    try:
        return send_from_directory('frames', filename)
    except FileNotFoundError:
        return "Image not found", 404

if __name__ == '__main__':
    app.run(debug=True)