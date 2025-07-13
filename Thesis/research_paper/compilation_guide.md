# راهنمای کامپایل مقاله

## پیش‌نیازها

برای کامپایل این مقاله به نرم‌افزارهای زیر نیاز دارید:

### LaTeX Distribution
- **Windows**: MiKTeX یا TeX Live
- **macOS**: MacTeX
- **Linux**: TeX Live (معمولاً از طریق package manager نصب می‌شود)

### فونت‌های فارسی
- **XB Nika**: برای متن فارسی
- **Times New Roman**: برای متن انگلیسی

### پکیج‌های مورد نیاز
مقاله از پکیج‌های زیر استفاده می‌کند:
- `xepersian` - برای پشتیبانی فارسی
- `geometry` - برای تنظیم صفحه
- `amsmath`, `amssymb` - برای فرمول‌های ریاضی
- `graphicx` - برای تصاویر
- `booktabs` - برای جداول
- `listings` - برای کد
- `algorithm`, `algorithmic` - برای الگوریتم‌ها
- `hyperref` - برای لینک‌ها
- `cite` - برای مراجع

## مراحل کامپایل

### روش اول: استفاده از XeLaTeX (توصیه شده)

```bash
# مرحله اول: کامپایل اولیه
xelatex MAGNET_Research_Paper.tex

# مرحله دوم: پردازش مراجع
xelatex MAGNET_Research_Paper.tex

# مرحله سوم: کامپایل نهایی (در صورت نیاز)
xelatex MAGNET_Research_Paper.tex
```

### روش دوم: استفاده از Makefile

```bash
make paper
```

## نکات مهم

1. **XeLaTeX**: حتماً از XeLaTeX استفاده کنید، نه از pdfLaTeX
2. **کدگذاری**: فایل با کدگذاری UTF-8 ذخیره شده است
3. **فونت‌ها**: در صورت عدم وجود فونت‌ها، آن‌ها را نصب کنید
4. **تصاویر**: در صورت وجود تصاویر، مسیر آن‌ها را بررسی کنید

## رفع مشکلات رایج

### خطای فونت
```
! Package fontspec Error: The font "XB Nika" cannot be found.
```
**راه‌حل**: فونت XB Nika را نصب کنید

### خطای پکیج
```
! LaTeX Error: File 'xepersian.sty' not found.
```
**راه‌حل**: پکیج xepersian را نصب کنید

### خطای کدگذاری
```
! Package inputenc Error: Unicode char
```
**راه‌حل**: از XeLaTeX استفاده کنید، نه از pdfLaTeX

## خروجی‌های تولیدی

پس از کامپایل موفق، فایل‌های زیر تولید می‌شوند:
- `MAGNET_Research_Paper.pdf` - فایل نهایی مقاله
- `MAGNET_Research_Paper.aux` - فایل کمکی
- `MAGNET_Research_Paper.log` - گزارش کامپایل
- `MAGNET_Research_Paper.out` - فایل bookmark ها

## بررسی کیفیت

پس از کامپایل موفق:
1. PDF تولیدی را باز کنید
2. فونت‌های فارسی و انگلیسی را بررسی کنید
3. فرمول‌های ریاضی را چک کنید
4. جداول و تصاویر را بررسی کنید
5. مراجع و ارجاعات را تایید کنید 