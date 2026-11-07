
import matplotlib.pyplot as plt
import os

def visualize_confidence(original_conf, attacked_conf, output_path='static/confidence_comparison.png'):
    labels = ['Before Attack', 'After Attack']
    scores = [original_conf, attacked_conf]
    colors = ['green', 'red']

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, scores, color=colors)
    plt.ylim(0, 100)
    plt.title('Prediction Confidence Comparison')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
