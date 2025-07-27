import plotly.express as px
import pandas as pd

def plot_pca_with_plotly(reduced, sentences):
    df = pd.DataFrame(reduced, columns=['x', 'y'])
    df['label'] = sentences

    fig = px.scatter(
        df, x='x', y='y', text='label',
        title="Vector Clustering in 2D space using PCA",
        width=800, height=600
    )
    fig.update_traces(textposition='top center', marker=dict(size=10))
    return fig
