# magnet_model


 python train_fast.py --percentage 100 --epochs 20                    ✔  11s  
Using device: cpu
Using device: cpu

======================================================================
 آموزش سریع مدل MAGNET روی 100% از داده‌ها 
======================================================================

پارامترهای آموزش سریع:
  درصد داده‌ها: 100% (بسیار کم برای توسعه سریع)
  تعداد اپوک‌ها: 20
  اندازه مدل: کوچک (حدود 200K پارامتر به جای 3.3M)

بارگذاری داده‌ها...
داده‌های پردازش شده با موفقیت بارگذاری شدند.
تعداد نمونه‌های آموزشی: 4641
تعداد نمونه‌های آزمایشی: 1451
نمونه‌های اولیه: 4641 آموزش, 1451 تست
نمونه‌های نهایی: 4641 آموزش, 1451 تست
مدل با 56,386 پارامتر آماده شد (حدود 0.06M)

--- شروع آموزش سریع برای 20 اپوک ---
Epoch [1/20] | Train Loss: 0.4477 | Val Acc: 0.9083 | Val F1: 0.9434
*** مدل جدید با F1=0.9434 ذخیره شد ***
Epoch [2/20] | Train Loss: 0.1732 | Val Acc: 0.9497 | Val F1: 0.9678
*** مدل جدید با F1=0.9678 ذخیره شد ***
Epoch [3/20] | Train Loss: 0.1042 | Val Acc: 0.9662 | Val F1: 0.9783
*** مدل جدید با F1=0.9783 ذخیره شد ***
Epoch [4/20] | Train Loss: 0.0814 | Val Acc: 0.9697 | Val F1: 0.9805
*** مدل جدید با F1=0.9805 ذخیره شد ***
Epoch [5/20] | Train Loss: 0.0668 | Val Acc: 0.9662 | Val F1: 0.9783
Epoch [6/20] | Train Loss: 0.0614 | Val Acc: 0.9697 | Val F1: 0.9804
Epoch [7/20] | Train Loss: 0.0540 | Val Acc: 0.9683 | Val F1: 0.9796
Epoch [8/20] | Train Loss: 0.0519 | Val Acc: 0.9704 | Val F1: 0.9809
*** مدل جدید با F1=0.9809 ذخیره شد ***
Epoch [9/20] | Train Loss: 0.0515 | Val Acc: 0.9690 | Val F1: 0.9800
Epoch [10/20] | Train Loss: 0.0505 | Val Acc: 0.9711 | Val F1: 0.9813
*** مدل جدید با F1=0.9813 ذخیره شد ***
Epoch [11/20] | Train Loss: 0.0493 | Val Acc: 0.9711 | Val F1: 0.9815
*** مدل جدید با F1=0.9815 ذخیره شد ***
Epoch [12/20] | Train Loss: 0.0500 | Val Acc: 0.9655 | Val F1: 0.9779
Epoch [13/20] | Train Loss: 0.0472 | Val Acc: 0.9704 | Val F1: 0.9809
Epoch [14/20] | Train Loss: 0.0465 | Val Acc: 0.9711 | Val F1: 0.9814
Epoch [15/20] | Train Loss: 0.0460 | Val Acc: 0.9690 | Val F1: 0.9800
Epoch [16/20] | Train Loss: 0.0427 | Val Acc: 0.9697 | Val F1: 0.9804
Epoch [17/20] | Train Loss: 0.0444 | Val Acc: 0.9704 | Val F1: 0.9810
Epoch [18/20] | Train Loss: 0.0448 | Val Acc: 0.9711 | Val F1: 0.9813
Epoch [19/20] | Train Loss: 0.0422 | Val Acc: 0.9704 | Val F1: 0.9810
Epoch [20/20] | Train Loss: 0.0486 | Val Acc: 0.9717 | Val F1: 0.9817
*** مدل جدید با F1=0.9817 ذخیره شد ***

زمان اجرا: 0:07:13.775182

ماتریس اغتشاش:
[[ 308   19]
 [  22 1102]]

نتایج نهایی:
  Accuracy: 0.9717
  Precision: 0.9831
  Recall: 0.9804
  F1 Score: 0.9817

مدل با موفقیت در models/fast_magnet_100percent.pth ذخیره شد.

//////////////////////////////////


    ~/Documents/final_magnet/magnet_model    develop !3 ?3     python train_final.py --embedding_dim 64 --num_heads 2 --num_layers 2 --batch_size 32 --data_percentage 50 --epochs 30 --ssl_weight 0
Using device: cpu
استفاده از دستگاه: cpu
تنظیمات در results/magnet_final_20250408_231645/config.json ذخیره شدند

======================================================================
 آموزش نهایی مدل MAGNET - 20250408_231645 
======================================================================

تنظیمات آموزش:
  embedding_dim: 64
  num_heads: 2
  num_layers: 2
  dim_feedforward: 256
  dropout: 0.3
  batch_size: 32
  learning_rate: 0.0005
  weight_decay: 0.01
  epochs: 30
  ssl_weight: 0.0
  data_percentage: 50
  patience: 7
  num_workers: 4
  class_weights: False

بارگذاری داده‌ها...
داده‌های پردازش شده با موفقیت بارگذاری شدند.
تعداد نمونه‌های آموزشی: 4641
تعداد نمونه‌های آزمایشی: 1451
نوع y_train: <class 'torch.Tensor'>
شکل نهایی y_train: (4641,)
تعداد نمونه‌های آموزشی: 4641
تعداد نمونه‌های آزمایشی: 1451
توزیع کلاس‌ها (آموزش): {np.int64(0): np.int64(1093), np.int64(1): np.int64(3548)}
استفاده از 50% داده‌ها: 2320 نمونه آموزشی

پیش‌پردازش و استانداردسازی داده‌ها...
ایجاد دیتاست‌ها و دیتالودرها...

آماده‌سازی مدل MAGNET...
تعداد کل پارامترها: 414,855
تعداد پارامترهای قابل آموزش: 408,455
/usr/lib/python3.13/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(

==================================================
 شروع آموزش مدل 
==================================================

Epoch 1/30
--------------------------------------------------
زمان اپوک: 165.14 ثانیه                                                                                                                    
Train Loss: 0.5507, Train Acc: 0.7608
Val Loss: 0.4663, Val Acc: 0.7988
Val Precision: 0.7946, Val Recall: 0.9982
Val F1: 0.8849, Val AUC: 0.7033
LR: 0.0005
ماتریس اغتشاش:
[[  37  290]
 [   2 1122]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.8849 ذخیره شد ***

Epoch 2/30
--------------------------------------------------
زمان اپوک: 159.98 ثانیه                                                                                                                    
Train Loss: 0.4564, Train Acc: 0.8224
Val Loss: 0.3774, Val Acc: 0.8739
Val Precision: 0.8822, Val Recall: 0.9662
Val F1: 0.9223, Val AUC: 0.8088
LR: 0.0005
ماتریس اغتشاش:
[[ 182  145]
 [  38 1086]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9223 ذخیره شد ***

Epoch 3/30
--------------------------------------------------
زمان اپوک: 160.67 ثانیه                                                                                                                    
Train Loss: 0.3832, Train Acc: 0.8582
Val Loss: 0.3215, Val Acc: 0.8918
Val Precision: 0.8884, Val Recall: 0.9840
Val F1: 0.9337, Val AUC: 0.8954
LR: 0.0005
ماتریس اغتشاش:
[[ 188  139]
 [  18 1106]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9337 ذخیره شد ***

Epoch 4/30
--------------------------------------------------
زمان اپوک: 160.32 ثانیه                                                                                                                    
Train Loss: 0.3297, Train Acc: 0.8871
Val Loss: 0.2617, Val Acc: 0.9118
Val Precision: 0.9157, Val Recall: 0.9760
Val F1: 0.9449, Val AUC: 0.9354
LR: 0.0005
ماتریس اغتشاش:
[[ 226  101]
 [  27 1097]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9449 ذخیره شد ***

Epoch 5/30
--------------------------------------------------
زمان اپوک: 158.27 ثانیه                                                                                                                    
Train Loss: 0.2932, Train Acc: 0.8931
Val Loss: 0.2353, Val Acc: 0.8925
Val Precision: 0.9636, Val Recall: 0.8950
Val F1: 0.9280, Val AUC: 0.9565
LR: 0.0005
ماتریس اغتشاش:
[[ 289   38]
 [ 118 1006]]

Epoch 6/30
--------------------------------------------------
زمان اپوک: 158.59 ثانیه                                                                                                                    
Train Loss: 0.2511, Train Acc: 0.9039
Val Loss: 0.2317, Val Acc: 0.9063
Val Precision: 0.9349, Val Recall: 0.9448
Val F1: 0.9398, Val AUC: 0.9595
LR: 0.0005
ماتریس اغتشاش:
[[ 253   74]
 [  62 1062]]

Epoch 7/30
--------------------------------------------------
زمان اپوک: 157.41 ثانیه                                                                                                                    
Train Loss: 0.2318, Train Acc: 0.9134
Val Loss: 0.1890, Val Acc: 0.9132
Val Precision: 0.9455, Val Recall: 0.9422
Val F1: 0.9439, Val AUC: 0.9707
LR: 0.0005
ماتریس اغتشاش:
[[ 266   61]
 [  65 1059]]

Epoch 8/30
--------------------------------------------------
زمان اپوک: 160.22 ثانیه                                                                                                                    
Train Loss: 0.2065, Train Acc: 0.9211
Val Loss: 0.1974, Val Acc: 0.9056
Val Precision: 0.9474, Val Recall: 0.9297
Val F1: 0.9385, Val AUC: 0.9691
LR: 0.0005
ماتریس اغتشاش:
[[ 269   58]
 [  79 1045]]

Epoch 9/30
--------------------------------------------------
زمان اپوک: 159.45 ثانیه                                                                                                                    
Train Loss: 0.1933, Train Acc: 0.9237
Val Loss: 0.1936, Val Acc: 0.9194
Val Precision: 0.9178, Val Recall: 0.9840
Val F1: 0.9498, Val AUC: 0.9768
LR: 0.00025
ماتریس اغتشاش:
[[ 228   99]
 [  18 1106]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9498 ذخیره شد ***

Epoch 10/30
--------------------------------------------------
زمان اپوک: 159.70 ثانیه                                                                                                                    
Train Loss: 0.1759, Train Acc: 0.9319
Val Loss: 0.1773, Val Acc: 0.9373
Val Precision: 0.9411, Val Recall: 0.9804
Val F1: 0.9603, Val AUC: 0.9802
LR: 0.00025
ماتریس اغتشاش:
[[ 258   69]
 [  22 1102]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9603 ذخیره شد ***

Epoch 11/30
--------------------------------------------------
زمان اپوک: 158.06 ثانیه                                                                                                                    
Train Loss: 0.1589, Train Acc: 0.9336
Val Loss: 0.1544, Val Acc: 0.9400
Val Precision: 0.9544, Val Recall: 0.9689
Val F1: 0.9616, Val AUC: 0.9825
LR: 0.00025
ماتریس اغتشاش:
[[ 275   52]
 [  35 1089]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9616 ذخیره شد ***

Epoch 12/30
--------------------------------------------------
زمان اپوک: 158.63 ثانیه                                                                                                                    
Train Loss: 0.1325, Train Acc: 0.9513
Val Loss: 0.1767, Val Acc: 0.9428
Val Precision: 0.9453, Val Recall: 0.9831
Val F1: 0.9638, Val AUC: 0.9854
LR: 0.00025
ماتریس اغتشاش:
[[ 263   64]
 [  19 1105]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9638 ذخیره شد ***

Epoch 13/30
--------------------------------------------------
زمان اپوک: 159.15 ثانیه                                                                                                                    
Train Loss: 0.1257, Train Acc: 0.9543
Val Loss: 0.1420, Val Acc: 0.9456
Val Precision: 0.9485, Val Recall: 0.9831
Val F1: 0.9655, Val AUC: 0.9879
LR: 0.00025
ماتریس اغتشاش:
[[ 267   60]
 [  19 1105]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9655 ذخیره شد ***

Epoch 14/30
--------------------------------------------------
زمان اپوک: 159.36 ثانیه                                                                                                                    
Train Loss: 0.1282, Train Acc: 0.9513
Val Loss: 0.1318, Val Acc: 0.9449
Val Precision: 0.9571, Val Recall: 0.9724
Val F1: 0.9647, Val AUC: 0.9886
LR: 0.00025
ماتریس اغتشاش:
[[ 278   49]
 [  31 1093]]

Epoch 15/30
--------------------------------------------------
زمان اپوک: 159.03 ثانیه                                                                                                                    
Train Loss: 0.1132, Train Acc: 0.9556
Val Loss: 0.1307, Val Acc: 0.9483
Val Precision: 0.9629, Val Recall: 0.9706
Val F1: 0.9668, Val AUC: 0.9893
LR: 0.00025
ماتریس اغتشاش:
[[ 285   42]
 [  33 1091]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9668 ذخیره شد ***

Epoch 16/30
--------------------------------------------------
زمان اپوک: 159.80 ثانیه                                                                                                                    
Train Loss: 0.1077, Train Acc: 0.9565
Val Loss: 0.1269, Val Acc: 0.9559
Val Precision: 0.9601, Val Recall: 0.9840
Val F1: 0.9719, Val AUC: 0.9906
LR: 0.00025
ماتریس اغتشاش:
[[ 281   46]
 [  18 1106]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9719 ذخیره شد ***

Epoch 17/30
--------------------------------------------------
زمان اپوک: 159.27 ثانیه                                                                                                                    
Train Loss: 0.1025, Train Acc: 0.9608
Val Loss: 0.1200, Val Acc: 0.9559
Val Precision: 0.9682, Val Recall: 0.9751
Val F1: 0.9716, Val AUC: 0.9906
LR: 0.00025
ماتریس اغتشاش:
[[ 291   36]
 [  28 1096]]

Epoch 18/30
--------------------------------------------------
زمان اپوک: 158.09 ثانیه                                                                                                                    
Train Loss: 0.1016, Train Acc: 0.9608
Val Loss: 0.1131, Val Acc: 0.9573
Val Precision: 0.9666, Val Recall: 0.9786
Val F1: 0.9726, Val AUC: 0.9909
LR: 0.00025
ماتریس اغتشاش:
[[ 289   38]
 [  24 1100]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9726 ذخیره شد ***

Epoch 19/30
--------------------------------------------------
زمان اپوک: 160.85 ثانیه                                                                                                                    
Train Loss: 0.0938, Train Acc: 0.9612
Val Loss: 0.1128, Val Acc: 0.9504
Val Precision: 0.9739, Val Recall: 0.9617
Val F1: 0.9678, Val AUC: 0.9913
LR: 0.00025
ماتریس اغتشاش:
[[ 298   29]
 [  43 1081]]

Epoch 20/30
--------------------------------------------------
زمان اپوک: 191.82 ثانیه                                                                                                                    
Train Loss: 0.0916, Train Acc: 0.9672
Val Loss: 0.1103, Val Acc: 0.9586
Val Precision: 0.9667, Val Recall: 0.9804
Val F1: 0.9735, Val AUC: 0.9913
LR: 0.00025
ماتریس اغتشاش:
[[ 289   38]
 [  22 1102]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9735 ذخیره شد ***

Epoch 21/30
--------------------------------------------------
زمان اپوک: 139.64 ثانیه                                                                                                                    
Train Loss: 0.0830, Train Acc: 0.9668
Val Loss: 0.1149, Val Acc: 0.9531
Val Precision: 0.9615, Val Recall: 0.9786
Val F1: 0.9700, Val AUC: 0.9913
LR: 0.00025
ماتریس اغتشاش:
[[ 283   44]
 [  24 1100]]

Epoch 22/30
--------------------------------------------------
زمان اپوک: 138.78 ثانیه                                                                                                                    
Train Loss: 0.0827, Train Acc: 0.9659
Val Loss: 0.1089, Val Acc: 0.9566
Val Precision: 0.9716, Val Recall: 0.9724
Val F1: 0.9720, Val AUC: 0.9909
LR: 0.00025
ماتریس اغتشاش:
[[ 295   32]
 [  31 1093]]

Epoch 23/30
--------------------------------------------------
زمان اپوک: 139.90 ثانیه                                                                                                                    
Train Loss: 0.0806, Train Acc: 0.9672
Val Loss: 0.1125, Val Acc: 0.9593
Val Precision: 0.9725, Val Recall: 0.9751
Val F1: 0.9738, Val AUC: 0.9918
LR: 0.00025
ماتریس اغتشاش:
[[ 296   31]
 [  28 1096]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9738 ذخیره شد ***

Epoch 24/30
--------------------------------------------------
زمان اپوک: 141.27 ثانیه                                                                                                                    
Train Loss: 0.0811, Train Acc: 0.9698
Val Loss: 0.1155, Val Acc: 0.9600
Val Precision: 0.9708, Val Recall: 0.9778
Val F1: 0.9743, Val AUC: 0.9916
LR: 0.00025
ماتریس اغتشاش:
[[ 294   33]
 [  25 1099]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9743 ذخیره شد ***

Epoch 25/30
--------------------------------------------------
زمان اپوک: 141.65 ثانیه                                                                                                                    
Train Loss: 0.0834, Train Acc: 0.9612
Val Loss: 0.1083, Val Acc: 0.9586
Val Precision: 0.9776, Val Recall: 0.9689
Val F1: 0.9732, Val AUC: 0.9919
LR: 0.00025
ماتریس اغتشاش:
[[ 302   25]
 [  35 1089]]

Epoch 26/30
--------------------------------------------------
زمان اپوک: 141.25 ثانیه                                                                                                                    
Train Loss: 0.0774, Train Acc: 0.9694
Val Loss: 0.1422, Val Acc: 0.9538
Val Precision: 0.9584, Val Recall: 0.9831
Val F1: 0.9706, Val AUC: 0.9909
LR: 0.00025
ماتریس اغتشاش:
[[ 279   48]
 [  19 1105]]

Epoch 27/30
--------------------------------------------------
زمان اپوک: 143.48 ثانیه                                                                                                                    
Train Loss: 0.0790, Train Acc: 0.9677
Val Loss: 0.1373, Val Acc: 0.9566
Val Precision: 0.9625, Val Recall: 0.9822
Val F1: 0.9723, Val AUC: 0.9894
LR: 0.00025
ماتریس اغتشاش:
[[ 284   43]
 [  20 1104]]

Epoch 28/30
--------------------------------------------------
زمان اپوک: 141.82 ثانیه                                                                                                                    
Train Loss: 0.0771, Train Acc: 0.9707
Val Loss: 0.1477, Val Acc: 0.9559
Val Precision: 0.9601, Val Recall: 0.9840
Val F1: 0.9719, Val AUC: 0.9892
LR: 0.00025
ماتریس اغتشاش:
[[ 281   46]
 [  18 1106]]

Epoch 29/30
--------------------------------------------------
زمان اپوک: 140.51 ثانیه                                                                                                                    
Train Loss: 0.0730, Train Acc: 0.9698
Val Loss: 0.1117, Val Acc: 0.9600
Val Precision: 0.9734, Val Recall: 0.9751
Val F1: 0.9742, Val AUC: 0.9915
LR: 0.000125
ماتریس اغتشاش:
[[ 297   30]
 [  28 1096]]

Epoch 30/30
--------------------------------------------------
زمان اپوک: 140.82 ثانیه                                                                                                                    
Train Loss: 0.0661, Train Acc: 0.9733
Val Loss: 0.1181, Val Acc: 0.9600
Val Precision: 0.9684, Val Recall: 0.9804
Val F1: 0.9744, Val AUC: 0.9915
LR: 0.000125
ماتریس اغتشاش:
[[ 291   36]
 [  22 1102]]
مدل در results/magnet_final_20250408_231645/magnet_best_model.pth ذخیره شد
*** بهترین مدل با F1=0.9744 ذخیره شد ***

==================================================
 ارزیابی نهایی مدل 
==================================================
Traceback (most recent call last):
  File "/home/alireza/Documents/final_magnet/magnet_model/train_final.py", line 667, in <module>
    main(args)
    ~~~~^^^^^^
  File "/home/alireza/Documents/final_magnet/magnet_model/train_final.py", line 579, in main
    model, _, _, _, _, _, _ = load_model(best_model_path, model)
                              ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/alireza/Documents/final_magnet/magnet_model/train_final.py", line 215, in load_model
    checkpoint = torch.load(model_path, map_location=device)
  File "/usr/lib/python3.13/site-packages/torch/serialization.py", line 1470, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([scalar])` or the `torch.serialization.safe_globals([scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
    ~/Doc/fi/magnet_model    develop !3 ?3                                                                     1 ✘  1h 17m 20s  

 /////////////////////////////////////???????/////



     ~/Doc/final_magnet/magnet_model    develop !4 ?3  python train_final.py --embedding_dim 64 --num_heads 2 --num_layers 2 --batch_size 32 --data_percentage 50 --epochs 30
Using device: cpu
استفاده از دستگاه: cpu
تنظیمات در results/magnet_final_20250409_005430/config.json ذخیره شدند

======================================================================
 آموزش نهایی مدل MAGNET - 20250409_005430 
======================================================================

تنظیمات آموزش:
  embedding_dim: 64
  num_heads: 2
  num_layers: 2
  dim_feedforward: 256
  dropout: 0.3
  batch_size: 32
  learning_rate: 0.0005
  weight_decay: 0.01
  epochs: 30
  ssl_weight: 0
  data_percentage: 50
  patience: 7
  num_workers: 4
  class_weights: False

بارگذاری داده‌ها...
داده‌های پردازش شده با موفقیت بارگذاری شدند.
تعداد نمونه‌های آموزشی: 4641
تعداد نمونه‌های آزمایشی: 1451
نوع y_train: <class 'torch.Tensor'>
شکل نهایی y_train: (4641,)
تعداد نمونه‌های آموزشی: 4641
تعداد نمونه‌های آزمایشی: 1451
توزیع کلاس‌ها (آموزش): {np.int64(0): np.int64(1093), np.int64(1): np.int64(3548)}
استفاده از 50% داده‌ها: 2320 نمونه آموزشی

پیش‌پردازش و استانداردسازی داده‌ها...
ایجاد دیتاست‌ها و دیتالودرها...

آماده‌سازی مدل MAGNET...
تعداد کل پارامترها: 414,855
تعداد پارامترهای قابل آموزش: 408,455
/usr/lib/python3.13/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(

==================================================
 شروع آموزش مدل 
==================================================

Epoch 1/30
--------------------------------------------------
زمان اپوک: 137.07 ثانیه                                                                                                                    
Train Loss: 0.5507, Train Acc: 0.7608
Val Loss: 0.4663, Val Acc: 0.7988
Val Precision: 0.7946, Val Recall: 0.9982
Val F1: 0.8849, Val AUC: 0.7033
LR: 0.0005
ماتریس اغتشاش:
[[  37  290]
 [   2 1122]]
*** بهترین مدل با F1=0.8849 ذخیره شد ***

Epoch 2/30
--------------------------------------------------
زمان اپوک: 139.54 ثانیه                                                                                                                    
Train Loss: 0.4564, Train Acc: 0.8224
Val Loss: 0.3774, Val Acc: 0.8739
Val Precision: 0.8822, Val Recall: 0.9662
Val F1: 0.9223, Val AUC: 0.8088
LR: 0.0005
ماتریس اغتشاش:
[[ 182  145]
 [  38 1086]]
*** بهترین مدل با F1=0.9223 ذخیره شد ***

Epoch 3/30
--------------------------------------------------
زمان اپوک: 138.21 ثانیه                                                                                                                    
Train Loss: 0.3832, Train Acc: 0.8582
Val Loss: 0.3215, Val Acc: 0.8918
Val Precision: 0.8884, Val Recall: 0.9840
Val F1: 0.9337, Val AUC: 0.8954
LR: 0.0005
ماتریس اغتشاش:
[[ 188  139]
 [  18 1106]]
*** بهترین مدل با F1=0.9337 ذخیره شد ***

Epoch 4/30
--------------------------------------------------
زمان اپوک: 138.13 ثانیه                                                                                                                    
Train Loss: 0.3297, Train Acc: 0.8871
Val Loss: 0.2617, Val Acc: 0.9118
Val Precision: 0.9157, Val Recall: 0.9760
Val F1: 0.9449, Val AUC: 0.9354
LR: 0.0005
ماتریس اغتشاش:
[[ 226  101]
 [  27 1097]]
*** بهترین مدل با F1=0.9449 ذخیره شد ***

Epoch 5/30
--------------------------------------------------
زمان اپوک: 136.38 ثانیه                                                                                                                    
Train Loss: 0.2932, Train Acc: 0.8931
Val Loss: 0.2353, Val Acc: 0.8925
Val Precision: 0.9636, Val Recall: 0.8950
Val F1: 0.9280, Val AUC: 0.9565
LR: 0.0005
ماتریس اغتشاش:
[[ 289   38]
 [ 118 1006]]

Epoch 6/30
--------------------------------------------------
زمان اپوک: 136.06 ثانیه                                                                                                                    
Train Loss: 0.2511, Train Acc: 0.9039
Val Loss: 0.2317, Val Acc: 0.9063
Val Precision: 0.9349, Val Recall: 0.9448
Val F1: 0.9398, Val AUC: 0.9595
LR: 0.0005
ماتریس اغتشاش:
[[ 253   74]
 [  62 1062]]

Epoch 7/30
--------------------------------------------------
زمان اپوک: 133.10 ثانیه                                                                                                                    
Train Loss: 0.2318, Train Acc: 0.9134
Val Loss: 0.1890, Val Acc: 0.9132
Val Precision: 0.9455, Val Recall: 0.9422
Val F1: 0.9439, Val AUC: 0.9707
LR: 0.0005
ماتریس اغتشاش:
[[ 266   61]
 [  65 1059]]

Epoch 8/30
--------------------------------------------------
زمان اپوک: 133.65 ثانیه                                                                                                                    
Train Loss: 0.2065, Train Acc: 0.9211
Val Loss: 0.1974, Val Acc: 0.9056
Val Precision: 0.9474, Val Recall: 0.9297
Val F1: 0.9385, Val AUC: 0.9691
LR: 0.0005
ماتریس اغتشاش:
[[ 269   58]
 [  79 1045]]

Epoch 9/30
--------------------------------------------------
زمان اپوک: 130.63 ثانیه                                                                                                                    
Train Loss: 0.1933, Train Acc: 0.9237
Val Loss: 0.1936, Val Acc: 0.9194
Val Precision: 0.9178, Val Recall: 0.9840
Val F1: 0.9498, Val AUC: 0.9768
LR: 0.00025
ماتریس اغتشاش:
[[ 228   99]
 [  18 1106]]
*** بهترین مدل با F1=0.9498 ذخیره شد ***

Epoch 10/30
--------------------------------------------------
زمان اپوک: 132.20 ثانیه                                                                                                                    
Train Loss: 0.1759, Train Acc: 0.9319
Val Loss: 0.1773, Val Acc: 0.9373
Val Precision: 0.9411, Val Recall: 0.9804
Val F1: 0.9603, Val AUC: 0.9802
LR: 0.00025
ماتریس اغتشاش:
[[ 258   69]
 [  22 1102]]
*** بهترین مدل با F1=0.9603 ذخیره شد ***

Epoch 11/30
--------------------------------------------------
زمان اپوک: 133.28 ثانیه                                                                                                                    
Train Loss: 0.1589, Train Acc: 0.9336
Val Loss: 0.1544, Val Acc: 0.9400
Val Precision: 0.9544, Val Recall: 0.9689
Val F1: 0.9616, Val AUC: 0.9825
LR: 0.00025
ماتریس اغتشاش:
[[ 275   52]
 [  35 1089]]
*** بهترین مدل با F1=0.9616 ذخیره شد ***

Epoch 12/30
--------------------------------------------------
زمان اپوک: 132.96 ثانیه                                                                                                                    
Train Loss: 0.1325, Train Acc: 0.9513
Val Loss: 0.1767, Val Acc: 0.9428
Val Precision: 0.9453, Val Recall: 0.9831
Val F1: 0.9638, Val AUC: 0.9854
LR: 0.00025
ماتریس اغتشاش:
[[ 263   64]
 [  19 1105]]
*** بهترین مدل با F1=0.9638 ذخیره شد ***

Epoch 13/30
--------------------------------------------------
زمان اپوک: 131.48 ثانیه                                                                                                                    
Train Loss: 0.1257, Train Acc: 0.9543
Val Loss: 0.1420, Val Acc: 0.9456
Val Precision: 0.9485, Val Recall: 0.9831
Val F1: 0.9655, Val AUC: 0.9879
LR: 0.00025
ماتریس اغتشاش:
[[ 267   60]
 [  19 1105]]
*** بهترین مدل با F1=0.9655 ذخیره شد ***

Epoch 14/30
--------------------------------------------------
زمان اپوک: 132.61 ثانیه                                                                                                                    
Train Loss: 0.1282, Train Acc: 0.9513
Val Loss: 0.1318, Val Acc: 0.9449
Val Precision: 0.9571, Val Recall: 0.9724
Val F1: 0.9647, Val AUC: 0.9886
LR: 0.00025
ماتریس اغتشاش:
[[ 278   49]
 [  31 1093]]

Epoch 15/30
--------------------------------------------------
زمان اپوک: 135.02 ثانیه                                                                                                                    
Train Loss: 0.1132, Train Acc: 0.9556
Val Loss: 0.1307, Val Acc: 0.9483
Val Precision: 0.9629, Val Recall: 0.9706
Val F1: 0.9668, Val AUC: 0.9893
LR: 0.00025
ماتریس اغتشاش:
[[ 285   42]
 [  33 1091]]
*** بهترین مدل با F1=0.9668 ذخیره شد ***

Epoch 16/30
--------------------------------------------------
زمان اپوک: 133.42 ثانیه                                                                                                                    
Train Loss: 0.1077, Train Acc: 0.9565
Val Loss: 0.1269, Val Acc: 0.9559
Val Precision: 0.9601, Val Recall: 0.9840
Val F1: 0.9719, Val AUC: 0.9906
LR: 0.00025
ماتریس اغتشاش:
[[ 281   46]
 [  18 1106]]
*** بهترین مدل با F1=0.9719 ذخیره شد ***

Epoch 17/30
--------------------------------------------------
زمان اپوک: 134.74 ثانیه                                                                                                                    
Train Loss: 0.1025, Train Acc: 0.9608
Val Loss: 0.1200, Val Acc: 0.9559
Val Precision: 0.9682, Val Recall: 0.9751
Val F1: 0.9716, Val AUC: 0.9906
LR: 0.00025
ماتریس اغتشاش:
[[ 291   36]
 [  28 1096]]

Epoch 18/30
--------------------------------------------------
زمان اپوک: 135.01 ثانیه                                                                                                                    
Train Loss: 0.1016, Train Acc: 0.9608
Val Loss: 0.1131, Val Acc: 0.9573
Val Precision: 0.9666, Val Recall: 0.9786
Val F1: 0.9726, Val AUC: 0.9909
LR: 0.00025
ماتریس اغتشاش:
[[ 289   38]
 [  24 1100]]
*** بهترین مدل با F1=0.9726 ذخیره شد ***

Epoch 19/30
--------------------------------------------------
زمان اپوک: 134.31 ثانیه                                                                                                                    
Train Loss: 0.0938, Train Acc: 0.9612
Val Loss: 0.1128, Val Acc: 0.9504
Val Precision: 0.9739, Val Recall: 0.9617
Val F1: 0.9678, Val AUC: 0.9913
LR: 0.00025
ماتریس اغتشاش:
[[ 298   29]
 [  43 1081]]

Epoch 20/30
--------------------------------------------------
زمان اپوک: 133.56 ثانیه                                                                                                                    
Train Loss: 0.0916, Train Acc: 0.9672
Val Loss: 0.1103, Val Acc: 0.9586
Val Precision: 0.9667, Val Recall: 0.9804
Val F1: 0.9735, Val AUC: 0.9913
LR: 0.00025
ماتریس اغتشاش:
[[ 289   38]
 [  22 1102]]
*** بهترین مدل با F1=0.9735 ذخیره شد ***

Epoch 21/30
--------------------------------------------------
زمان اپوک: 134.04 ثانیه                                                                                                                    
Train Loss: 0.0830, Train Acc: 0.9668
Val Loss: 0.1149, Val Acc: 0.9531
Val Precision: 0.9615, Val Recall: 0.9786
Val F1: 0.9700, Val AUC: 0.9913
LR: 0.00025
ماتریس اغتشاش:
[[ 283   44]
 [  24 1100]]

Epoch 22/30
--------------------------------------------------
زمان اپوک: 133.54 ثانیه                                                                                                                    
Train Loss: 0.0827, Train Acc: 0.9659
Val Loss: 0.1089, Val Acc: 0.9566
Val Precision: 0.9716, Val Recall: 0.9724
Val F1: 0.9720, Val AUC: 0.9909
LR: 0.00025
ماتریس اغتشاش:
[[ 295   32]
 [  31 1093]]

Epoch 23/30
--------------------------------------------------
زمان اپوک: 135.15 ثانیه                                                                                                                    
Train Loss: 0.0806, Train Acc: 0.9672
Val Loss: 0.1125, Val Acc: 0.9593
Val Precision: 0.9725, Val Recall: 0.9751
Val F1: 0.9738, Val AUC: 0.9918
LR: 0.00025
ماتریس اغتشاش:
[[ 296   31]
 [  28 1096]]
*** بهترین مدل با F1=0.9738 ذخیره شد ***

Epoch 24/30
--------------------------------------------------
زمان اپوک: 138.00 ثانیه                                                                                                                    
Train Loss: 0.0811, Train Acc: 0.9698
Val Loss: 0.1155, Val Acc: 0.9600
Val Precision: 0.9708, Val Recall: 0.9778
Val F1: 0.9743, Val AUC: 0.9916
LR: 0.00025
ماتریس اغتشاش:
[[ 294   33]
 [  25 1099]]
*** بهترین مدل با F1=0.9743 ذخیره شد ***

Epoch 25/30
--------------------------------------------------
زمان اپوک: 139.81 ثانیه                                                                                                                    
Train Loss: 0.0834, Train Acc: 0.9612
Val Loss: 0.1083, Val Acc: 0.9586
Val Precision: 0.9776, Val Recall: 0.9689
Val F1: 0.9732, Val AUC: 0.9919
LR: 0.00025
ماتریس اغتشاش:
[[ 302   25]
 [  35 1089]]

Epoch 26/30
--------------------------------------------------
زمان اپوک: 137.72 ثانیه                                                                                                                    
Train Loss: 0.0774, Train Acc: 0.9694
Val Loss: 0.1422, Val Acc: 0.9538
Val Precision: 0.9584, Val Recall: 0.9831
Val F1: 0.9706, Val AUC: 0.9909
LR: 0.00025
ماتریس اغتشاش:
[[ 279   48]
 [  19 1105]]

Epoch 27/30
--------------------------------------------------
زمان اپوک: 137.10 ثانیه                                                                                                                    
Train Loss: 0.0790, Train Acc: 0.9677
Val Loss: 0.1373, Val Acc: 0.9566
Val Precision: 0.9625, Val Recall: 0.9822
Val F1: 0.9723, Val AUC: 0.9894
LR: 0.00025
ماتریس اغتشاش:
[[ 284   43]
 [  20 1104]]

Epoch 28/30
--------------------------------------------------
زمان اپوک: 136.64 ثانیه                                                                                                                    
Train Loss: 0.0771, Train Acc: 0.9707
Val Loss: 0.1477, Val Acc: 0.9559
Val Precision: 0.9601, Val Recall: 0.9840
Val F1: 0.9719, Val AUC: 0.9892
LR: 0.00025
ماتریس اغتشاش:
[[ 281   46]
 [  18 1106]]

Epoch 29/30
--------------------------------------------------
زمان اپوک: 137.81 ثانیه                                                                                                                    
Train Loss: 0.0730, Train Acc: 0.9698
Val Loss: 0.1117, Val Acc: 0.9600
Val Precision: 0.9734, Val Recall: 0.9751
Val F1: 0.9742, Val AUC: 0.9915
LR: 0.000125
ماتریس اغتشاش:
[[ 297   30]
 [  28 1096]]

Epoch 30/30
--------------------------------------------------
زمان اپوک: 138.86 ثانیه                                                                                                                    
Train Loss: 0.0661, Train Acc: 0.9733
Val Loss: 0.1181, Val Acc: 0.9600
Val Precision: 0.9684, Val Recall: 0.9804
Val F1: 0.9744, Val AUC: 0.9915
LR: 0.000125
ماتریس اغتشاش:
[[ 291   36]
 [  22 1102]]
*** بهترین مدل با F1=0.9744 ذخیره شد ***

==================================================
 ارزیابی نهایی مدل 
==================================================
                                                                                                                                           
نتایج نهایی (بهترین مدل از اپوک 30):
دقت: 0.9600
Precision: 0.9684
Recall: 0.9804
F1 Score: 0.9744
AUC: 0.9915

ماتریس اغتشاش نهایی:
[[ 291   36]
 [  22 1102]]
نمودارها در results/magnet_final_20250409_005430/training_curves.png ذخیره شدند

زمان کل اجرا: 1:14:06.684365
مدل نهایی و خروجی‌ها در results/magnet_final_20250409_005430 ذخیره شدند
    ~/Doc/fi/magnet_model    develop !4 ?3                                                                       ✔  1h 14m 12s  