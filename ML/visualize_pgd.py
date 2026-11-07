import matplotlib.pyplot as plt
import os

def visualize_pgd_accuracy(teacher_acc, student_acc, output_path='static/pgd_accuracy.png'):
    labels = ['Teacher Model', 'Student Model']
    scores = [teacher_acc, student_acc]
    colors = ['blue', 'orange']

    plt.figure(figsize=(5, 4))
    plt.bar(labels, scores, color=colors)
    plt.ylim(0, 100)
    plt.title('PGD Attack - Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
