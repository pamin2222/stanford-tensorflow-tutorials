# Step 6: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne_thai.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontname='Garuda')

    plt.savefig(filename)


try:
    from sklearn.manifold import TSNE
    import matplotlib
    from matplotlib import rcParams

    matplotlib.rc('font', family='Garuda')
    import matplotlib.pyplot as plt

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Garuda']

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    print(labels);
    print(low_dim_embs);
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
