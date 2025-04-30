#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
دانلود و پردازش دیتاست AndroZoo
این اسکریپت یک زیرمجموعه از دیتاست AndroZoo را دانلود و پردازش می‌کند
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

def download_androzoo_metadata(api_key, sample_size=1000):
    """دانلود متادیتای نمونه‌ها از AndroZoo"""
    print("دریافت لیست نمونه‌ها از AndroZoo...")
    
    # URL برای دریافت لیست نمونه‌ها
    url = "https://androzoo.uni.lu/api/download"
    params = {
        "apikey": api_key,
        "limit": sample_size,
        "sort": "dex_date",
        "order": "desc"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"خطا در دریافت داده‌ها: {response.status_code}")
    
    return response.json()

def process_sample(sha256, api_key):
    """پردازش یک نمونه از AndroZoo"""
    # URL برای دریافت اطلاعات نمونه
    url = f"https://androzoo.uni.lu/api/analyze/{sha256}"
    params = {"apikey": api_key}
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None
    
    data = response.json()
    
    # استخراج ویژگی‌های استاتیک
    static_features = {
        'sha256': sha256,
        'permissions': len(data.get('permissions', [])),
        'activities': len(data.get('activities', [])),
        'services': len(data.get('services', [])),
        'receivers': len(data.get('receivers', [])),
        'providers': len(data.get('providers', [])),
        'api_calls': len(data.get('api_calls', [])),
        'intent_filters': len(data.get('intent_filters', [])),
        'hardware_components': len(data.get('hardware_components', [])),
        'label': 1 if data.get('vt_detection', 0) > 0 else 0
    }
    
    # استخراج ویژگی‌های دینامیک
    dynamic_features = {
        'sha256': sha256,
        'network_connections': len(data.get('network_connections', [])),
        'system_calls': len(data.get('system_calls', [])),
        'file_operations': len(data.get('file_operations', [])),
        'memory_usage': data.get('memory_usage', 0),
        'cpu_usage': data.get('cpu_usage', 0)
    }
    
    return static_features, dynamic_features

def main():
    """تابع اصلی"""
    # دریافت API Key از کاربر
    api_key = input("لطفا API Key خود را از AndroZoo وارد کنید: ")
    
    # تعداد نمونه‌ها
    sample_size = 1000
    
    # دانلود متادیتا
    metadata = download_androzoo_metadata(api_key, sample_size)
    
    # ایجاد لیست‌های خالی برای ذخیره داده‌ها
    static_features_list = []
    dynamic_features_list = []
    
    # پردازش هر نمونه
    print("\nپردازش نمونه‌ها...")
    for sample in tqdm(metadata):
        try:
            static, dynamic = process_sample(sample['sha256'], api_key)
            if static and dynamic:
                static_features_list.append(static)
                dynamic_features_list.append(dynamic)
            time.sleep(1)  # برای جلوگیری از rate limiting
        except Exception as e:
            print(f"خطا در پردازش نمونه {sample['sha256']}: {str(e)}")
    
    # تبدیل به DataFrame
    static_df = pd.DataFrame(static_features_list)
    dynamic_df = pd.DataFrame(dynamic_features_list)
    
    # ذخیره داده‌ها
    print("\nذخیره داده‌ها...")
    os.makedirs("data/androzoo", exist_ok=True)
    
    static_df.to_csv("data/androzoo/static_features.csv", index=False)
    dynamic_df.to_csv("data/androzoo/dynamic_features.csv", index=False)
    
    # ایجاد فایل labels
    labels_df = static_df[['sha256', 'label']]
    labels_df.to_csv("data/androzoo/labels.csv", index=False)
    
    print("\nپردازش داده‌ها با موفقیت انجام شد!")
    print(f"تعداد نمونه‌های پردازش شده: {len(static_df)}")

if __name__ == "__main__":
    main() 