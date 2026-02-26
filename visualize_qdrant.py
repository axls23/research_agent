import os
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sklearn.manifold import TSNE

# Load env variables
load_dotenv()
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

print("Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Fetch all vectors
print("Scrolling through Qdrant to retrieve all chunks...")
points = client.scroll(
    collection_name="research_entities",
    limit=2000,
    with_payload=True,
    with_vectors=True,
)[0]

print(f"Retrieved {len(points)} entities. Reducing 384 dimensions to 2D with t-SNE...")

import numpy as np

# Extract layout data
vectors = np.array([p.vector for p in points])
payloads = [p.payload for p in points]

# Reduce dimensionality
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create DataFrame for Plotly
df = pd.DataFrame(
    {
        "x": reduced_vectors[:, 0],
        "y": reduced_vectors[:, 1],
        "entity": [p["text"] for p in payloads],
        "label": [p.get("label", "Concept") for p in payloads],
        "paper": [p.get("paper_ids", ["Unknown"])[0] for p in payloads],
    }
)

print("Generating HTML Visualization...")
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="label",
    hover_name="entity",
    hover_data=["paper"],
    title="Semantic Space Map of Research Entities (t-SNE scaled BGE-small embeddings)",
)

# Dark theme layout mapping
fig.update_layout(template="plotly_dark", title_font_size=20, height=800, width=1400)

# Save to html
output_file = "qdrant_semantic_map.html"
fig.write_html(output_file)
print(f"Done! Open {os.path.abspath(output_file)} in your web browser.")
