import pandas as pd
import cudf, cuml

df = pd.read_csv("data/data.csv")

columns=['name', 'artists', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
df_mod = df[columns]

keys = df_mod.iloc[:,:2].values.tolist()
features = df_mod.iloc[:,2:].to_numpy()
features = (features-features.min())/(features.max()-features.min())


df = cudf.DataFrame(features)
embed = cuml.UMAP(n_neighbors=20, n_epochs=100, 
                min_dist=0.1, init='spectral').fit_transform(df)
np_embed = embed.to_pandas().to_numpy()

np.save("result/embeddings.npy", np_embed)