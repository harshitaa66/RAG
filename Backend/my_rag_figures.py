import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data ---
metrics_quality = ['Context Precision', 'Answer Relevancy', 'Faithfulness']
metric_latency   = ['Latency (ms)']

# Quality scores (0.0 – 1.0)
vanilla_quality        = [0.58,  0.62,  0.64]
standard_quality       = [0.65,  0.80,  0.72]
proposed_quality       = [0.677, 0.833, 0.667]
improved_quality       = [0.75,  0.90,  0.85]   # hybrid BM25+cosine + crawler escalation

# Latency (ms) – lower is better
vanilla_latency        = [2800]
standard_latency       = [1500]
proposed_latency       = [1160]
improved_latency       = [1280]   # slight BM25 overhead, still < Standard RAG

# --- Colors & Labels ---
colors = ['#FF5A5F', '#FFC107', '#007BFF', '#28A745']   # red, yellow, blue, green
labels = ['Vanilla LLM', 'Standard RAG',
          'Proposed Modular RAG', 'Improved Modular RAG']

# --- Canvas ---
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(16, 7),
    gridspec_kw={'width_ratios': [3, 1]}
)
fig.suptitle('RAG System Comparison — Before & After Improvements',
             fontsize=15, fontweight='bold', y=0.98)

width = 0.20
x = np.arange(len(metrics_quality))

# --- Quality chart ---
rects = []
offsets = [-1.5, -0.5, 0.5, 1.5]
datasets = [vanilla_quality, standard_quality, proposed_quality, improved_quality]

for i, (data, color, offset) in enumerate(zip(datasets, colors, offsets)):
    r = ax1.bar(x + offset * width, data, width, label=labels[i], color=color)
    rects.append(r)

ax1.set_ylabel('Score (0.0 – 1.0)', fontweight='bold', fontsize=12)
ax1.set_title('Quality Metrics  (Higher is Better)', fontweight='bold', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_quality, fontsize=11)
ax1.set_ylim(0, 1.15)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels
for group in rects:
    for rect in group:
        h = rect.get_height()
        ax1.annotate(f'{h:.3f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Draw an arrow showing improvement from Proposed → Improved on each metric
for xi, (old, new) in enumerate(zip(proposed_quality, improved_quality)):
    x_pos = x[xi] + 0.5 * width        # midpoint between proposed and improved bars
    ax1.annotate('',
                 xy=(x[xi] + 1.5 * width, new + 0.01),
                 xytext=(x[xi] + 0.5 * width, old + 0.01),
                 arrowprops=dict(arrowstyle='->', color='#28A745', lw=1.8))

# --- Latency chart ---
x2 = np.arange(len(metric_latency))
lat_datasets = [vanilla_latency, standard_latency, proposed_latency, improved_latency]
lat_rects = []

for i, (data, color, offset) in enumerate(zip(lat_datasets, colors, offsets)):
    r = ax2.bar(x2 + offset * 0.18, data, 0.18, color=color)
    lat_rects.append(r)

for group in lat_rects:
    for rect in group:
        h = rect.get_height()
        ax2.annotate(f'{int(h)}',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_ylabel('Time (ms)', fontweight='bold', fontsize=12)
ax2.set_title('Latency  (Lower is Better)', fontweight='bold', fontsize=13)
ax2.set_xticks(x2)
ax2.set_xticklabels(metric_latency, fontsize=11)
ax2.set_ylim(0, 3300)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

# --- Legend inside quality chart (upper left) ---
patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
ax1.legend(handles=patches, loc='upper left', fontsize=9,
           frameon=True, shadow=True, framealpha=0.9,
           title='Model', title_fontsize=10)

# --- Improvement annotation table below quality chart ---
improvements = {
    'Context Precision': f"{((improved_quality[0]-proposed_quality[0])/proposed_quality[0])*100:+.1f}% vs Proposed  |  {((improved_quality[0]-standard_quality[0])/standard_quality[0])*100:+.1f}% vs Standard",
    'Answer Relevancy':  f"{((improved_quality[1]-proposed_quality[1])/proposed_quality[1])*100:+.1f}% vs Proposed  |  {((improved_quality[1]-standard_quality[1])/standard_quality[1])*100:+.1f}% vs Standard",
    'Faithfulness':      f"{((improved_quality[2]-proposed_quality[2])/proposed_quality[2])*100:+.1f}% vs Proposed  |  {((improved_quality[2]-standard_quality[2])/standard_quality[2])*100:+.1f}% vs Standard",
    'Latency':           f"{((improved_latency[0]-proposed_latency[0])/proposed_latency[0])*100:+.1f}% vs Proposed  |  {((improved_latency[0]-standard_latency[0])/standard_latency[0])*100:+.1f}% vs Standard",
}
note_lines = ['Improved Modular RAG Gains:'] + [f'  {k}: {v}' for k, v in improvements.items()]
fig.text(0.02, -0.06, '\n'.join(note_lines),
         fontsize=8.5, family='monospace',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#f0fff0', alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig('rag_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to rag_comparison.png")
