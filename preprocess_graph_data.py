from torch_geometric.data import Data
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data.data import DataEdgeAttr

# اضافه کردن DataEdgeAttr به لیست global های مجاز
add_safe_globals([DataEdgeAttr])

# بارگذاری داده‌های گرافی با weights_only=False
graph_data = torch.load('processed_data/graph_data.pt', weights_only=False)

# بررسی داده‌های گرافی
print("بررسی داده‌های گرافی:")
print(f"تعداد گره‌ها: {graph_data.num_nodes}")
print(f"تعداد یال‌ها: {graph_data.num_edges}")
print(f"ابعاد ویژگی‌های گره‌ها: {graph_data.x.size()}")
print(f"ابعاد ویژگی‌های یال‌ها: {graph_data.edge_attr.size()}")
print(f"برچسب‌ها: {graph_data.y.size()}")

# اضافه کردن batch برای پردازش گراف به صورت یکجا
graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)  # همه گره‌ها توی یک batch

# ذخیره داده‌های گرافی پیش‌پردازش‌شده
torch.save(graph_data, 'processed_data/graph_data_processed.pt')

print("داده‌های گرافی پیش‌پردازش و ذخیره شدند.")