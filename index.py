import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de visualização
sns.set(style="whitegrid")

# Caminhos ajustados para o dataset
data_path = 'data/UCI HAR Dataset/'
features_path = os.path.join(data_path, 'features.txt')
x_train_path = os.path.join(data_path, 'train/X_train.txt')
x_test_path = os.path.join(data_path, 'test/X_test.txt')
y_train_path = os.path.join(data_path, 'train/y_train.txt')
y_test_path = os.path.join(data_path, 'test/y_test.txt')

# Checagem de arquivos
required_files = [features_path, x_train_path, x_test_path, y_train_path, y_test_path]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Arquivo não encontrado: {file}. Verifique os caminhos.")

# 2. Carregar os dados
features = pd.read_csv(features_path, sep=r'\s+', header=None, names=['index', 'feature'])
print(f"Total de features carregadas: {features.shape[0]}")

# Carregando os dados de treino e teste
X_train = pd.read_csv(x_train_path, sep=r'\s+', header=None)
X_test = pd.read_csv(x_test_path, sep=r'\s+', header=None)
y_train = pd.read_csv(y_train_path, header=None)
y_test = pd.read_csv(y_test_path, header=None)

# Concatenar os dados de treino e teste
X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)

# 3. Pré-processamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Redução de dimensionalidade (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"Variância explicada pelos dois primeiros componentes: {explained_variance:.2f}")

# 5. Escolha do número ideal de clusters (Método do Cotovelo)
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# Gráfico do Método do Cotovelo (Inércia)
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Inércia")
plt.show()

# Gráfico de Dispersão dos Componentes PCA (Variância Explicada)
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=10)
plt.title("Distribuição dos Dados - PCA (2 componentes)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label='Atividade')
plt.show()

# 6. Aplicar o K-means com o valor de K escolhido
optimal_k = 4  # Escolha o K baseado no gráfico do método do cotovelo
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)
clusters = kmeans.fit_predict(X_pca)

# 7. Avaliar a qualidade dos clusters (Silhouette Score)
silhouette_avg = silhouette_score(X_pca, clusters)
print(f"Silhouette Score para K={optimal_k}: {silhouette_avg:.2f}")

# 8. Visualização dos Clusters
plt.figure(figsize=(8, 5))
for cluster in range(optimal_k):
    plt.scatter(X_pca[clusters == cluster, 0], X_pca[clusters == cluster, 1], label=f"Cluster {cluster + 1}")

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c="red", marker="X", label="Centroids")
plt.title(f"Clusters para K={optimal_k}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()
