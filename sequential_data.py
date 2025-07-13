import numpy as np
import torch
import pandas as pd

# بارگذاری برچسب‌ها
y = pd.read_csv('processed_data/y.csv')

# تعریف واژگان API (فقط ویژگی‌هایی که API یا کلاس هستن)
feature_names = pd.read_csv('dataset/drebin.csv').drop(columns=['class']).columns.tolist()
api_vocab = [i for i, name in enumerate(feature_names) if '.' in name or '/' in name]
api_names = [name for name in feature_names if '.' in name or '/' in name]
print(f"تعداد APIها در واژگان: {len(api_vocab)}")

# تعریف طول توالی (مثلاً هر اپلیکیشن 20 فراخوانی API داره)
seq_length = 20

# اگه فقط یک نمونه داری، نمی‌تونیم توالی معنی‌داری بسازیم
if len(y) < 2:
    print("برای تولید داده‌های متوالی حداقل به دو اپلیکیشن نیاز داریم. لطفاً داده‌های بیشتری فراهم کنید.")
else:
    # تولید توالی‌ها
    num_apps = len(y)
    seq_data = []

    for i in range(num_apps):
        if y.iloc[i].values[0] == 1:  # بدافزار
            # بدافزارها بیشتر APIهای مشکوک رو فراخوانی می‌کنن
            # فرض می‌کنیم APIهای مشکوک اونایی هستن که توی بدافزارها بیشتر ظاهر می‌شن
            suspicious_apis = [idx for idx, name in zip(api_vocab, api_names) if name in [
                'android.telephony.SmsManager', 'SEND_SMS', 'RECEIVE_SMS', 'WRITE_SMS', 'Runtime.exec', 
                'DexClassLoader', 'System.loadLibrary', 'TelephonyManager.getDeviceId', 'mount', 'remount'
            ]]
            seq = np.random.choice(suspicious_apis, size=seq_length, replace=True)
        else:  # بی‌خطر
            # بی‌خطرها بیشتر APIهای معمولی رو فراخوانی می‌کنن
            normal_apis = [idx for idx in api_vocab if idx not in suspicious_apis]
            seq = np.random.choice(normal_apis, size=seq_length, replace=True)
        seq_data.append(seq)

    # تبدیل به تنسور
    seq_data = np.array(seq_data)  # Convert list of arrays to a single NumPy array
    seq_data = torch.LongTensor(seq_data)  # Then convert to a PyTorch tensor

    # ذخیره داده‌های متوالی
    torch.save(seq_data, 'processed_data/seq_data.pt')

    print("داده‌های متوالی تولید و ذخیره شدند.")
    print(f"تعداد نمونه‌ها: {seq_data.size(0)}")
    print(f"طول هر توالی: {seq_data.size(1)}")