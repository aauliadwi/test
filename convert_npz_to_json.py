import numpy as np
import json

# Load .npz file
data = np.load("app/item_embeddings.npz", allow_pickle=True)
item_ids = data["food_ids"]
titles = data["titles"]
embeddings = data["embeddings"]

# Konversi ke format JSON
items = []
for i in range(len(item_ids)):
    items.append({
        "food_id": item_ids[i],          # langsung string, tanpa konversi ke int
        "title": str(titles[i]),
        "embedding": embeddings[i].tolist()
    })

# Simpan ke file JSON
with open("app/item_embeddings.json", "w") as f:
    json.dump(items, f)
