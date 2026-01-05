"""
uv pip install ollama tqdm
wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/gemma3-pro/gt.csv

uv run src/create_prediction_gemma3.py
"""

import ollama
import csv
from tqdm import tqdm

system_message = """Given the following Hebrew sentence, convert it to IPA phonemes.
Input Format: A Hebrew sentence.
Output Format: A string of IPA phonemes.
"""

with open("gt.csv", "r", encoding="utf-8") as f, open("pred_gemma3.csv", "w", encoding="utf-8") as f_pred:
    reader = csv.DictReader(f)
    writer = csv.writer(f_pred)
    writer.writerow(["id", "phonemes"])
    
    for row in tqdm(reader):
        id = row["id"]
        transcript = row["transcript"]
        
        # Run inference with ollama
        response = ollama.chat(
            model="gemma3-g2p",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcript}
            ],
            options={
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 64,
                "num_predict": 150,
                "stop": ["<end_of_turn>", "</s>"]
            }
        )
        
        phonemes = response["message"]["content"].strip()
        writer.writerow([id, phonemes])

print("\nPredictions saved to pred_gemma3.csv")