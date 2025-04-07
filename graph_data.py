import networkx as nx
import numpy as np
from torch_geometric.data import Data
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# بارگذاری داده‌های جدولی
X_tabular = pd.read_csv('processed_data/X_tabular.csv')
y = pd.read_csv('processed_data/y.csv')

# پیش‌پردازش داده‌ها: جایگزینی مقادیر '?' با NaN و سپس پر کردن با میانگین
X_tabular = X_tabular.replace('?', np.nan)
# تبدیل همه ستون‌ها به نوع عددی
X_tabular = X_tabular.apply(pd.to_numeric, errors='coerce')
# پر کردن مقادیر NaN با میانگین هر ستون
X_tabular = X_tabular.fillna(X_tabular.mean())

# اگه فقط یک نمونه داری، نمی‌تونیم گراف بسازیم
if len(X_tabular) < 2:
    print("برای ساخت گراف حداقل به دو اپلیکیشن نیاز داریم. لطفاً داده‌های بیشتری فراهم کنید.")
else:
    # ساخت گراف
    G = nx.Graph()

    # اضافه کردن گره‌ها (هر اپلیکیشن یک گره)
    num_apps = len(X_tabular)
    for i in range(num_apps):
        G.add_node(i, features=X_tabular.iloc[i].values, label=y.iloc[i].values[0])

    # تعریف یال‌ها بر اساس شباهت
    similarity_matrix = cosine_similarity(X_tabular)
    threshold = 0.8  # آستانه برای تعریف یال

    edge_index = []
    edge_attr = []
    for i in range(num_apps):
        for j in range(i + 1, num_apps):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])
                edge_index.append([i, j])
                edge_attr.append([similarity_matrix[i, j]])

    # تبدیل به فرمت PyTorch Geometric
    node_features = torch.FloatTensor(X_tabular.values)
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attr)
    labels = torch.LongTensor(y.values.flatten())

    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)

    # ذخیره داده‌های گرافی
    torch.save(graph_data, 'processed_data/graph_data.pt')

    print("داده‌های گرافی تولید و ذخیره شدند.")
    print(f"تعداد گره‌ها: {graph_data.num_nodes}")
    print(f"تعداد یال‌ها: {graph_data.num_edges}")