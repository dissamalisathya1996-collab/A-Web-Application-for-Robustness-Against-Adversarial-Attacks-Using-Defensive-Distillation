import matplotlib.pyplot as plt
import os

def visualize_fgsm_accuracy(teacher_acc, student_acc, output_path='static/fgsm_accuracy.png'):
    labels = ['Teacher Model', 'Student Model']
    scores = [teacher_acc, student_acc]
    colors = ['red', 'green']

    plt.figure(figsize=(5, 4))
    plt.bar(labels, scores, color=colors)
    plt.ylim(0, 100)
    plt.title('FGSM Attack - Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
