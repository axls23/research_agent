import os
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.decomposition import PCA
import plotly.express as px

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER') or os.getenv('NEO4J_USERNAME')
pwd = os.getenv('NEO4J_PASSWORD')
db = os.getenv('NEO4J_DATABASE')

if not (uri and user and pwd):
    raise SystemExit('Missing Neo4j connection env vars')

driver = GraphDatabase.driver(uri, auth=(user, pwd))
rows = []
with driver.session(database=db) as s:
    result = s.run(
        """
        MATCH (n:PRISMAEntity)
        WHERE n.embedding IS NOT NULL
        RETURN n.text AS text,
               coalesce(n.prisma_label, n.label, 'unknown') AS label,
               n.embedding AS embedding
        LIMIT 5000
        """
    )
    for r in result:
        emb = r['embedding']
        if isinstance(emb, list) and len(emb) > 2:
            rows.append({
                'text': (r['text'] or '')[:240],
                'label': r['label'] or 'unknown',
                'embedding': emb,
            })

driver.close()

if not rows:
    raise SystemExit('No PRISMAEntity embeddings found in Neo4j.')

X = np.array([r['embedding'] for r in rows], dtype=float)
labels = [r['label'] for r in rows]
texts = [r['text'] for r in rows]

pca = PCA(n_components=2, random_state=42)
XY = pca.fit_transform(X)

plot_df = pd.DataFrame({
    'x': XY[:, 0],
    'y': XY[:, 1],
    'label': labels,
    'text': texts,
})

fig = px.scatter(
    plot_df,
    x='x',
    y='y',
    color='label',
    hover_data=['text'],
    title='Neo4j PRISMA Entity Vector Map (PCA 2D)',
)
fig.update_layout(template='plotly_dark', height=840, width=1400)

out_path = os.path.join('outputs', 'neo4j_vector_map.html')
fig.write_html(out_path)
print(f'POINTS={len(rows)}')
print(f'OUTPUT={out_path}')
