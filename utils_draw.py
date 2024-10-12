import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_sentiment_distribution(post):
  pca = PCA(n_components=2)
  post = pca.fit_transform(post)
  
  for i in range(post.shape[0]):
    plt.scatter(post[i, 0], post[i, 1], c='green', alpha=0.6, edgecolors='w', s=50)
    plt.text(post[i, 0], post[i, 1], f'{i}', fontsize=9, ha='right', va='bottom')
  plt.title('Sentiment Distribution of Posts')
  plt.show()

def visualize_state_space(data1, data2, type):
    tsne = TSNE(n_components=2, random_state=42)
    combined_data = np.vstack((data1, data2))
    tsne_result = tsne.fit_transform(combined_data)

    # Create a 2D scatter plot
    fig, ax = plt.subplots()

    # Define colors
    colors = ['red'] * data1.shape[0] + ['green'] * data2.shape[0]

    # Scatter plot
    ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors)
    ax.set_title(f"Emotional State of {type} Users")

    # Create custom legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Without S-SSM')
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='With S-SSM')
    ax.legend(handles=[red_patch, green_patch])
    plt.show()


