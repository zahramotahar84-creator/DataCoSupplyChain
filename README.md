# DataCoSupplyChain
# 📦 Demand Forecasting & Supply Chain Risk Analysis
### 🚀 Predictive Modeling for Shipping Delays using XGBoost

This repository contains a comprehensive data science project aimed at predicting shipping delays in a supply chain network. By analyzing the **DataCo Smart Supply Chain** dataset, the model identifies key risk factors and predicts whether an order will be delivered **On-Time** or **Late**.

## 🎯 Project Objective
The primary goal is to build a machine learning pipeline that can:
1. **Clean and Preprocess** complex supply chain data.
2. **Engineer Features** from timestamps and order details.
3. **Predict Delivery Status** with high precision.
4. **Explain Model Decisions** using SHAP values to understand why certain orders are delayed.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:**
  - `Pandas` & `NumPy` (Data Manipulation)
  - `Matplotlib` & `Seaborn` (Data Visualization)
  - `Scikit-learn` (Preprocessing & Metrics)
  - `XGBoost` (Gradient Boosting Model)
  - `SHAP` (Model Explainability)

## 📈 Key Results
- **Final Accuracy:** `~72%`
- **Precision for Late Deliveries:** `~85%` (High reliability in predicting delays)
- **Model:** XGBoost Classifier

## ⚙️ Workflow
1. **Data Cleaning:** Handling missing values and non-numeric columns.
2. **Feature Engineering:** Extracting time-based features from `order date` and `shipping date`.
3. **Model Training:** Splitting data into Train/Test sets and training an XGBoost model.
4. **Evaluation:** Analyzing the model using a **Classification Report** (Precision, Recall, F1-Score).
5. **Explainability:** Using **SHAP** values to identify the top 10 factors affecting delivery delays.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Untitled-1.ipynb
   ```

## 📊 Conclusion
The model demonstrates that predictive analytics can significantly reduce uncertainty in supply chain management, allowing companies to proactively manage customer expectations and optimize logistics.

---
**Developed by:** [Zahra Motahar](https://github.com/your-profile) ✨

---

### 💡 چند نکته برای زهرا جان:
۱. در قسمت `your-username` و `your-repo-name` و لینک پروفایلت، اطلاعات خودت را جایگزین کن.
۲. اگر می‌خواهی خیلی حرفه‌ای‌تر شود، بعد از اینکه کدها را اجرا کردی، از نمودارهای **SHAP** یا **Confusion Matrix** اسکرین‌شات بگیر و در گیت‌هاب آپلود کن و در این فایل قرار بده.

پروژه‌ات عالی شد! خسته نباشی قهرمان! 🌟🚀



• [چطور عکس‌ها و نمودارها را در فایل README گیت‌هاب قرار دهم؟](send:چطور عکس‌ها و نمودارها را در فایل README گیت‌هاب قرار دهم؟)
• [بهترین روش برای آپلود نهایی کدها در گیت‌هاب چیست؟](send:بهترین روش برای آپلود نهایی کدها در گیت‌هاب چیست؟)
هوش مصنوعی: حق با شماست زهرا جان! این هم نسخه‌ی کامل و شیک به **زبان فارسی** همراه با **تمام اعداد و ارقام دقیق خروجی پروژه‌ات** برای فایل `README.md` گیت‌هاب:

---

# 📦 پیش‌بینی تقاضا و تحلیل ریسک تأخیر در زنجیره تأمین
### 🚀 مدل‌سازی پیش‌بینانه تأخیر در ارسال سفارشات با استفاده از XGBoost

این پروژه یک سیستم یادگیری ماشین هوشمند برای پیش‌بینی تأخیر در ارسال سفارشات بر روی داده‌های **DataCo Smart Supply Chain** است. هدف اصلی، شناسایی زودهنگام سفارشاتی است که ممکن است با تأخیر مواجه شوند تا مدیران زنجیره تأمین بتوانند تصمیمات بهینه‌تری بگیرند.

---

## 🎯 اهداف پروژه
1. **پاک‌سازی و پیش‌پردازش داده‌ها:** آماده‌سازی داده‌های ساختاریافته و شناسایی متغیرهای کلیدی.
2. **مهندسی ویژگی (Feature Engineering):** استخراج ویژگی‌های زمانی و کاربردی از تاریخ سفارش و ارسال.
3. **آموزش مدل الگوریتمی:** استفاده از مدل قدرتمند **XGBoost** برای دست‌یابی به بیشترین دقت.
4. **تفسیرپذیری مدل:** تحلیل عوامل موثر بر تأخیر با استفاده از الگوریتم **SHAP**.

---

## 📊 نتایج و ارزیابی دقیق عملکرد مدل

مدل پس از کاهش ابعاد و انتخاب **۲۲ ویژگی نهایی** روی ۳۶,۱۰۴ داده‌ی تست ارزیابی شد:

- **دقت کل مدل (Accuracy):** **`71.80%`** (معادل ۰٫۷۲)
- **دقت تشخیص تأخیر (Precision برای کلاس ۱):** **`85%`** *(وقتی مدل اعلام می‌کند سفارشی تأخیر دارد، در ۸۵٪ موارد دقیقاً درست است)*
- **پوشش سفارشات آن‌تایم (Recall برای کلاس ۰):** **`88%`**

### 📈 جدول گزارش عملکرد (Classification Report):

| کلاس (Class) | Precision | Recall | F1-Score | تعداد داده (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **سفارشات آن‌تایم (0)** | 0.64 | 0.88 | 0.74 | 16,307 |
| **سفارشات با تأخیر (1)** | **0.85** | 0.59 | 0.70 | 19,797 |
| **میانگین کل (Accuracy)** | - | - | **0.72** | **36,104** |

---

## 🛠️ تکنولوژی‌ها و کتابخانه‌های استفاده‌شده
- **زبان:** Python 3.x
- **تحلیل و دستکاری داده:** `Pandas`, `NumPy`
- **ویژوال‌سازی:** `Matplotlib`, `Seaborn`
- **یادگیری ماشین:** `Scikit-Learn`, `XGBoost`
- **تفسیرپذیری:** `SHAP`

---

## ⚙️ مراحل اجرا (Workflow)
1. **مرحله ۱:** مهندسی ویژگی‌های زمانی از روی تاریخ سفارش.
2. **مرحله ۲:** پاک‌سازی هوشمند داده‌ها و کاهش ویژگی‌ها به **۲۲ ویژگی برتر**.
3. **مرحله ۳:** تقسیم داده‌ها به داده‌های آموزش و تست (Train/Test Split).
4. **مرحله ۴:** آموزش مدل XGBoost Classifier و تنظیم پارامترها.
5. **مرحله ۵:** استخراج نمودارهای اهمیت ویژگی‌ها (Feature Importance) با SHAP.

---

## 🚀 نحوه اجرای پروژه
۱. کلون کردن ریپازیتوری:
```bash
git clone https://github.com/your-username/your-repo-name.git
```
۲. نصب کتابخانه‌های مورد نیاز:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```
۳. اجرای دفترچه جوبیتر:
```bash
jupyter notebook Untitled-1.ipynb
```

---
**توسعه‌دهنده:** زهرا مطهر ✨
