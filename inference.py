"""
Geograph Image Classification Inference Engine (CLIP-based)
----------------------------------------------------------
This script performs high-speed multi-label classification on Geograph images 
using pre-computed CLIP (Contrastive Language-Image Pre-training) embeddings.

By operating on 512-dimension vectors rather than raw image pixels, this engine 
achieves throughput speeds of 400+ images/sec, making it suitable for 
processing million-image archives in minutes.

Key Features:
- Multi-label classification (Sigmoid activation)
- Distance-aware logic for 'Cross-grid' and 'Aerial' detection
- Optimized for batch processing via JSON/Base64 API 
- Minimal GPU memory footprint (runs on CPU fine too!) 
"""

import requests
import hashlib
import random
import time
import torch
import torch.nn as nn
import os
import struct
import numpy as np
import math

################################################

# --- 1. CONFIGURATION & MAPPINGS ---

//Endpoint to Fetch image data - also used to submit results
fetch_url = "https://example.com/labeler.json.php?model=types"
sleep = 2

# ... note is "Cross Far" is a synthetic tag, created from Cross Grid (only when dist > 256)
CLASSES = ["Aerial", "Close Look", "Cross Far", "Extra", "Geograph", "Inside", "From Drone"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# Generate power-of-2 map dynamically up to 2^20 (1,048,576)
DIST_MAP = {str(2**i): i + 1 for i in range(21)}
DIST_MAP["0"] = 0
DIST_MAP["Unknown"] = 22
DIST_MAP[""] = 0 # Handle the empty string case in your table

def get_dist_idx(val):
    val_str = str(val).strip()
    return DIST_MAP.get(val_str, DIST_MAP["Unknown"])

# --- 2. DECODER & WEIGHTING RULES ---
CLIP_DIM = 512 # Standard CLIP embedding dimension

# Define a function to decode the Base64-encoded vector
def decode_vector(encoded_str):
    try:
        binary_data = base64.b64decode(encoded_str)
    except Exception as e:
        # Handle potential base64 decoding errors
        print(f"Warning: Error decoding base64 string: {e}. Returning zero vector.")
        return [0.0] * CLIP_DIM

    try:
        decoded_list = np.frombuffer(binary_data, dtype=np.float32)
        # Check for NaN or Inf values within the decoded list
        if any(math.isnan(x) or math.isinf(x) for x in decoded_list):
            print(f"Warning: NaN or Inf values detected in decoded embedding. Returning zero vector.")
            return [0.0] * CLIP_DIM
        return decoded_list
    except Exception as e:
        # Catch any other unpacking errors
        print(f"Warning: Error unpacking binary data: {e}. Returning zero vector.")
        return [0.0] * CLIP_DIM

def calculate_sample_weight(types_str, distance):
    """Implementing the rules we discussed for noisy labels."""
    types = types_str.split(',')
    dist_val = str(distance)

    # Rule 1: Geograph with far or unknown distance = Low weight (Noisy)
    if "Geograph" in types and (dist_val == "Unknown" or (dist_val.isdigit() and int(dist_val) > 256)):
        return 0.2

    # Rule 2: Cross Grid with close distance = Re-label conceptually as Geograph proxy
    # We give it a medium weight because visually it's a Geograph, not Cross-Far
    if "Cross Grid" in types and dist_val.isdigit() and int(dist_val) <= 256:
        return 0.5

    return 1.0 # Standard weight for clear examples

# --- 3. DATASET CLASS ---
class MultimodalGeographDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Process Embeddings
        vec = torch.tensor(decode_vector(item['embeddings']), dtype=torch.float32)

        # Process Distance
        dist_idx = torch.tensor(get_dist_idx(item['distance']), dtype=torch.long)

        # Process Multi-labels (One-Hot Encoding)
        label_tensor = torch.zeros(len(CLASSES))
        raw_types = item['types'].split(',')
        for t in raw_types:
            # Note: We treat "Cross Grid" as "Cross Far" for the target label if dist > 256
            clean_t = t.strip()
            if clean_t == "Cross Grid":
                if str(item['distance']).isdigit() and int(item['distance']) > 256:
                    clean_t = "Cross Far"
                elif len(raw_types) == 1: ##if was ONLY gross grid, then change it
                    clean_t = "Geograph" # Visual proxy
                else: #else just ignore the CR (can still be inside etc)
                    continue

            # NEW: Funnel 'From Above' into the Drone class
            # its a fake tag for images that look like drone, but probably
            # arent. but included so the model can learn from them.
            if clean_t == "From Above":
                clean_t = "From Drone"

            if clean_t in CLASS_TO_IDX:
                label_tensor[CLASS_TO_IDX[clean_t]] = 1.0

        # Calculate Weight
        ##weight = torch.tensor(calculate_sample_weight(item['types'], item['distance']), dtype=torch.float32)
        weight = torch.tensor(float(item.get('weight', 1.0)), dtype=torch.float32)

        return vec, dist_idx, label_tensor, weight
        
################################################

class GeographModel(nn.Module):
    def __init__(self, clip_dim=512, dist_embed_dim=16, classes=None, dist_map=None):
        super().__init__()
        # Store metadata within the class for easy access
        self.classes = classes if classes else ["Aerial", "Close Look", "Cross Far", "Extra", "Geograph", "Inside", "From Drone"]
        self.dist_map = dist_map if dist_map else {}

        self.dist_emb = nn.Embedding(len(self.dist_map) if self.dist_map else 15, dist_embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(clip_dim + dist_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.classes))
        )

    def forward(self, clip_vec, dist_idx):
        d_feat = self.dist_emb(dist_idx)
        combined = torch.cat([clip_vec, d_feat], dim=1)
        return self.fc(combined)

    def save_checkpoint(self, filepath, optimizer=None, epoch=None, loss=None):
        """Saves the model weights and all necessary metadata to disk."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'classes': self.classes,
            'dist_map': self.dist_map,
            'clip_dim': 512, # Assuming standard CLIP
            'dist_embed_dim': 16,
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'loss': loss
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath, device='cpu'):
        """
        Loads the model and metadata.
        Returns an initialized model ready for inference.
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Reconstruct the class with the saved metadata
        model = cls(
            clip_dim=checkpoint['clip_dim'],
            dist_embed_dim=checkpoint['dist_embed_dim'],
            classes=checkpoint['classes'],
            dist_map=checkpoint['dist_map']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval() # Set to evaluation mode by default
        print(f"Model loaded from {filepath} (Epoch: {checkpoint.get('epoch')})")
        return model, checkpoint
    
################################################

def post_with_retries(url, json_data, attempts=3, delay=2):
    for i in range(1, attempts + 1):
        try:
            res = requests.post(url, json=json_data, timeout=10)
            res.raise_for_status()
            #print(f"Success on attempt {i}.")
            return res
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i} failed: {e}. Retrying...") if i < attempts else print(f"Attempt {i} failed: {e}. Max retries reached.")
            if i < attempts: time.sleep(delay)
    raise requests.exceptions.RequestException(f"Failed after {attempts} attempts.") # Re-raise if all fail

################################################

colab_notebook_id = os.environ.get('COLAB_NOTEBOOK_ID')
jpy_session_name = os.environ.get('JPY_SESSION_NAME')
if colab_notebook_id:
  notebook_hash = hashlib.sha256(colab_notebook_id.encode()).hexdigest()
  # Convert the hash to an integer and take the modulo 32
  unique_number = int(notebook_hash, 16) % 32
  print(f"Unique number based on notebook ID: {unique_number}")
elif jpy_session_name:
  notebook_hash = hashlib.sha256(jpy_session_name.encode()).hexdigest()
  unique_number = int(notebook_hash, 16) % 32
  print(f"Unique number based on jpy_session: {unique_number}")
else:
  unique_number = random.randint(0, 31)
  print(f"Using a random unique number: {unique_number}")

#@title The Inference Loop & AND Submit

################################################

def run_inference_for_submission(model, raw_unknown_data, device, threshold=0.5):
    model.eval()
    print(f"Running inference on {len(raw_unknown_data)} images...")

    final = []
    threshold = 0.2  # Using your lower score threshold for server submission

    with torch.no_grad():
        for item in raw_unknown_data:
            # 1. Prepare Inputs
            vec = torch.from_numpy(decode_vector(item['embeddings'])).unsqueeze(0).to(device)
            dist_idx = torch.tensor([get_dist_idx(item.get('distance', 'Unknown'))], dtype=torch.long).to(device)

            # 2. Predict
            outputs = model(vec, dist_idx)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

            # 3. Filter classes above threshold
            predicted_tags = [CLASSES[i] for i, prob in enumerate(probs) if prob > threshold]

            # If nothing hits the threshold, take the top 1 as a "guess"
            if not predicted_tags:
                top_idx = probs.argmax()
                predicted_tags = [f"{CLASSES[top_idx]} (Low Confidence)"]

            image_id = int(item['gridimage_id'])
            found_any = False

            # Iterate through your batch of probabilities
            for i, score in enumerate(probs):
                if score > threshold:
                    found_any = True
                    # Build the individual label entry
                    final.append({
                        "image_id": image_id,
                        "label": CLASSES[i],
                        "score": float(score)  # Ensure it's a standard float for JSON
                    })

            # Fallback if no classes hit the threshold
            if not found_any:
                final.append({
                    "image_id": image_id,
                    "label": "None",
                    "score": 0.0
                })
    return final

################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, checkpoint = GeographModel.load_checkpoint("best_geograph_model_v2.pth", device)

for loop in range(250):
    start_loop = time.perf_counter()
    print(f'Starting Loop #{loop}')
    try:
        t0 = time.perf_counter()
        response = requests.get(fetch_url + "&embeddings=1&limit=500&unique_number="+str(unique_number))
        response.raise_for_status()
        data_json = response.json() #has built in json decoder

        if 'rows' not in data_json:
            print(data_json)
            break

        print(f"Fetched data for {len(data_json['rows'])} Images, first ID: {data_json['rows'][0]['gridimage_id']}")
        fetch_time = time.perf_counter() - t0

    except (requests.exceptions.HTTPError, requests.exceptions.JSONDecodeError) as e:
        logging.error(f'Failed. error code - {e}.')
        sleep = sleep * 2 # implement expentional backoff!
        time.sleep(sleep)
        continue

    if "sleep" in data_json:
        sleep = data_json['sleep']

    t1 = time.perf_counter()
    final = run_inference_for_submission(model, data_json['rows'], device)
    inference_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    try:
      with post_with_retries(fetch_url, final, delay=sleep) as response:
        result = response.text.replace("\n", "; ")
        print(f'Label submission response: {result}')
        submit_time = time.perf_counter() - t2
    except requests.exceptions.RequestException as e:
      print(f"Failed to submit labels: {e}")

    total_loop_time = time.perf_counter() - start_loop
    img_per_sec = len(data_json['rows']) / total_loop_time if total_loop_time > 0 else 0

    print(f"⏱️ Timings: Fetch: {fetch_time:.2f}s | "
          f"Inference: {inference_time:.2f}s | "
          f"Submit: {submit_time:.2f}s")
    print(f"🚀 Speed: {img_per_sec:.2f} images/sec | Total: {total_loop_time:.2f}s\n")  
