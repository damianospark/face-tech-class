"""
Generate an HTML report for fi-cv-demo: load embeddings, compute distance, make 3D PCA scatter,
and create an HTML grouping of step images for enroll and probe.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent
ENROLL_DIR = BASE / 'output-enroll'
PROBE_DIR = BASE / 'output-chk'
REPORT_DIR = BASE / 'report'
REPORT_DIR.mkdir(exist_ok=True)

# helper load embedding
def load_embedding(p):
    if not p.exists():
        return None
    return np.load(p)

# collect embeddings
emb_en = load_embedding(ENROLL_DIR / '05.20_enroll_embedding.npy')
emb_pr = load_embedding(PROBE_DIR / '05.20_probe_embedding.npy')
if emb_en is None or emb_pr is None:
    print('Embeddings not found; run demo.py first')
    raise SystemExit(1)

# To visualize vectors from the origin in 3D, we need at least 3 points for PCA.
# We add the origin (a zero vector) as the third point.
origin = np.zeros_like(emb_en)
X = np.vstack([emb_en, emb_pr, origin])

pca = PCA(n_components=3)
Y = pca.fit_transform(X)

# plot arrows from the transformed origin to each embedding point in 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# The PCA-transformed origin is Y[2]
origin_3d = Y[2]

# Draw vectors from the transformed origin
enroll_vec = Y[0] - origin_3d
probe_vec = Y[1] - origin_3d
ax.quiver(origin_3d[0], origin_3d[1], origin_3d[2], enroll_vec[0], enroll_vec[1], enroll_vec[2], color='blue', label='enroll', arrow_length_ratio=0.1)
ax.quiver(origin_3d[0], origin_3d[1], origin_3d[2], probe_vec[0], probe_vec[1], probe_vec[2], color='red', label='probe', arrow_length_ratio=0.1)

# Adjust view angle and zoom to better visualize the two vectors
ax.view_init(elev=20, azim=30) # Adjust elevation and azimuth
all_points = np.vstack([Y[0], Y[1], origin_3d])
max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), all_points[:,1].max()-all_points[:,1].min(), all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
mid = np.mean(all_points, axis=0)
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

ax.legend()
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA of Embedding Vectors')
plt.savefig(REPORT_DIR / 'report_3d.png')
plt.close(fig)

# compute Euclidean distance in original embedding space
from numpy.linalg import norm
edist = float(norm(emb_en - emb_pr))
cos = float(np.dot(emb_en, emb_pr) / (norm(emb_en)*norm(emb_pr) + 1e-12))

# build html page grouping artifacts
html = ["<html><head><meta charset='utf-8'><title>fi-cv-demo report</title></head><body>"]
html.append(f"<h1>fi-cv-demo report</h1>")
html.append(f"<p>Euclidean distance: {edist:.6f} &nbsp; Cosine similarity: {cos:.6f}</p>")
# show 3d plot
html.append(f"<h2>3D PCA projection</h2><img src='report/report_3d.png' style='max-width:600px;'>")
# helper to list steps
steps = [
    ('01_detection.jpg', 'Detection'),
    ('02_aligned.jpg', 'Aligned'),
    ('03_preprocessed.jpg', 'Preprocessed'),
    ('04_aligned_parts_overlay.jpg', 'Parts overlay'),
]
part_steps = [
    (f'05.{i:02d}_{part}.jpg', part) for i, part in enumerate(['left_eye','right_eye','nose','mouth'], start=1)
]
all_steps = steps + part_steps

html.append('<h2>Artifacts</h2>')
html.append('<table border=1 cellpadding=6 cellspacing=0 style="width:100%;">')
html.append('<thead><tr style="text-align:center;"><th>Step</th><th>Enroll</th><th>Probe</th></tr></thead>')
html.append('<tbody>')

# 3-column layout
for fname, label in all_steps:
    epath = ENROLL_DIR.relative_to(BASE) / fname
    ppath = PROBE_DIR.relative_to(BASE) / fname
    max_width = '160px' if '05.' in fname else '320px'
    
    html.append('<tr style="text-align:center;">')
    html.append(f'<td><b>{label}</b></td>')
    html.append(f"<td><img src='{epath}' style='max-width:{max_width};'></td>")
    html.append(f"<td><img src='{ppath}' style='max-width:{max_width};'></td>")
    html.append('</tr>')

html.append('</tbody></table>')

# embeddings visual & text
html.append('<h2>Embeddings</h2>')
html.append(f"<p>Enroll embedding length: {len(emb_en)}; Probe length: {len(emb_pr)}</p>")
html.append(f"<p>Euclidean distance: {edist:.6f} &nbsp; Cosine similarity: {cos:.6f}</p>")

html.append('</body></html>')

open(REPORT_DIR / 'report.html', 'w').write('\n'.join(html))
print('Report generated at', REPORT_DIR / 'report.html')
