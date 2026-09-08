import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import xgboost as xgb
# ۱. بارگذاری داده (مطمئن شو مسیر فایل درست است)
df = pd.read_csv("C:\\Users\\user\\Downloads\\DataCoSupplyChainDataset.csv", encoding='latin1')
# ۲. بررسی ابعاد (چند تا ردیف و ستون داریم؟)
print(f"Dimensions: {df.shape}")
# ۳. لیست ستون‌ها (ببین چقدر زیاد و شلوغ هستند!)
print(df.columns)
# ۴. بررسی وضعیت فاجعه (ستون‌هایی که مقدار خالی دارند)
null_counts = df.isnull().sum()
print("\n--- Missing Values ---")
print(null_counts[null_counts > 0])
# ۵. نگاهی به چند ردیف اول
print(df.head())
# ۱. حذف ستون‌هایی که اطلاعات تکراری دارند یا کاملاً خالی هستند
cols_to_drop = [
'Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname',
'Product Description', 'Order Zipcode', 'Customer Zipcode'
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
# ۲. مدیریت مقادیر خالی (Handling Missing Values)
# در DataCo معمولاً ستون Customer Country یا Order State مقادیر خالی دارند
df['Customer Country'] = df['Customer Country'].fillna("Unknown")
# ۳. تبدیل تاریخ‌ها (بسیار حیاتی برای زنجیره تأمین)
df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
df['shipping_date'] = pd.to_datetime(df['shipping date (DateOrders)'])
print("پاکسازی اولیه انجام شد. ابعاد جدید:", df.shape)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# انتخاب ستون‌های عددی مهم
num_cols = ['Sales', 'Order Item Quantity', 'Order Item Total', 'Product Price']
# Scaling (تبدیل به میانگین صفر و واریانس یک)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
# Normalization (آوردن مقادیر بین ۰ و ۱ برای الگوریتم‌های حساس)
# معمولاً برای شبکه‌های عصبی از MinMaxScaler استفاده می‌کنیم
nm_scaler = MinMaxScaler()
df['Days for shipping (real)'] = nm_scaler.fit_transform(df[['Days for shipping (real)']])
print("اسکیلینگ و نرمال‌سازی با موفقیت انجام شد.")
# ۱. ساخت ویژگی‌های زمانی (بسیار مهم برای پیش‌بینی فروش)
df['order_month'] = df['order_date'].dt.month
df['order_day_of_week'] = df['order_date'].dt.dayofweek
df['order_hour'] = df['order_date'].dt.hour
# ۲. محاسبه اختلاف زمان (برای تحلیل ریسک تأخیر)
# تفاوت بین زمانی که قول دادیم و زمانی که واقعاً فرستادیم
df['delivery_gap'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
# ۳. برچسب‌گذاری هدف ریسک (اگر مثبت باشد یعنی تأخیر داشتیم)
df['is_late'] = (df['delivery_gap'] > 0).astype(int)
# ۴. میانگین فروش هر دسته (Target Encoding ساده برای تقویت پیش‌بینی فروش)
cat_sales_map = df.groupby('Category Name')['Sales'].mean().to_dict()
df['category_avg_sales'] = df['Category Name'].map(cat_sales_map)
print("ویژگی‌های استراتژیک ساخته شدند.")
print(df[['order_month', 'delivery_gap', 'is_late','Sales']].head())
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 6))
# محور اول: میانگین فروش در ماه‌های مختلف
ax1 = sns.lineplot(data=df, x='order_month', y='Sales', color='blue', label='Average Sales', marker='o')
ax1.set_ylabel('Sales', color='blue')
# محور دوم: نرخ تأخیر در همان ماه‌ها
ax2 = ax1.twinx()
sns.lineplot(data=df, x='order_month', y='is_late', color='red', label='Late Risk Rate', ax=ax2, marker='s')
ax2.set_ylabel('Late Delivery Risk', color='red')
plt.title('Relationship between Sales Volume and Delivery Risk')
plt.show()
# ۱. تبدیل متغیرهای متنی به عدد (Label Encoding برای ستون‌های دسته‌بندی شده)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# ستون‌هایی که باید عددی شوند
cat_cols = ['Type', 'Delivery Status', 'Category Name', 'Customer City', 'Shipping Mode']
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
# ۲. جدا کردن ویژگی‌ها (X) از هدف (y)
# فرض کن فعلاً می‌خواهیم ریسک تاخیر را پیش‌بینی کنیم
X = df.drop(['is_late', 'Sales', 'order_date', 'shipping_date'], axis=1)
y_risk = df['is_late'] # هدف اول: ریسک تاخیر
y_sales = df['Sales'] # هدف دوم: میزان فروش
print("داده‌ها آماده ورود به مدل شدند. تعداد ویژگی‌ها:", X.shape[1])
# ۱. شناسایی ستون‌هایی که هنوز عدد نیستند
non_numeric_cols = X.select_dtypes(exclude=['number']).columns
print("ستون‌های غیرعددی پیدا شده:", non_numeric_cols)
# ۲. حذف ستون‌های غیرعددی باقی‌مانده (مثل تاریخ‌های اصلی که مدل نمی‌فهمد)
X_clean = X.select_dtypes(include=['number'])
# ۳. تقسیم مجدد داده‌ها با دیتای کاملاً عددی
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_risk, test_size=0.2, random_state=42)
# ۴. حالا اجرای دوباره XGBoost
model_xgb = xgb.XGBClassifier(
n_estimators=100,
learning_rate=0.1,
max_depth=6,
objective='binary:logistic',
random_state=42
)
model_xgb.fit(X_train, y_train)
# ۵. پیش‌بینی
y_pred = model_xgb.predict(X_test)
print("\nهورا! مدل با موفقیت آموزش دید.")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(df.columns.tolist())
def run_supply_chain_analysis(df):
    """
    اجرای تحلیل کامل پیش‌بینی تأخیر در ارسال بدون تقلب داده‌ای
    """
    
    print("--- مرحله ۱: مهندسی ویژگی‌های زمانی ---")
    # استخراج اطلاعات از ستون تاریخ سفارش
    df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
    df['order_month'] = df['order_date'].dt.month
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    df['order_hour'] = df['order_date'].dt.hour
    
    # ۲. تعیین هدف (Target)
    y = df['Late_delivery_risk']
    
    print("--- مرحله ۲: پاکسازی هوشمند داده‌ها ---")
    # حذف تمام ستون‌هایی که نباید در مدل باشند (شخصی، IDها، و تقلب‌ها)
    black_list = [
        # ستون‌های نشت داده (Data Leakage) - مواردی که در عکستان باعث تقلب شده بود
        'is_late', 'Late_delivery_risk', 'Delivery Status', 'Order Status', 
        'Days for shipping (real)', 'delivery_gap', 'Sales',
        
        # IDها و شناسه‌های عددی (بدون ارزش یادگیری)
        'Order Customer Id', 'Order Item Cardprod Id', 'Customer Id', 'Order Id', 
        'Product Card Id', 'Order Item Id', 'Category Id', 'Department Id', 
        'Order Customer Key', 'Customer Zipcode', 'Order Zipcode',
        
        # اطلاعات شخصی و متنی (Privacy & Text)
        'Customer Email', 'Customer Fname', 'Customer Lname', 'Customer Password', 
        'Customer Street', 'Customer City', 'Customer State', 'Customer Country',
        'Product Name', 'Product Image', 'Order City', 'Order State', 'Order Country',
        
        # ستون‌های زمانی خام
        'order date (DateOrders)', 'shipping date (DateOrders)', 'order_date'
    ]
    
    # انتخاب ویژگی‌ها (فقط عددی‌ها منهای لیست سیاه)
    X = df.select_dtypes(include=['number']).drop(columns=black_list, errors='ignore')
    print(f"✅ تعداد ویژگی‌های نهایی برای آموزش: {X.shape[1]}")
    # ۳. تقسیم داده‌ها به آموزش و تست
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("--- مرحله ۳: آموزش مدل XGBoost ---")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    # ۴. ارزیابی مدل
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"🎯 دقت نهایی مدل: {accuracy:.4f}")
    print("="*40)
    print("\n📋 گزارش عملکرد دقیق:")
    print(classification_report(y_test, y_pred))
    
    # ۵. بصری‌سازی (نمودار اهمیت ویژگی‌ها)
    plt.figure(figsize=(10, 8))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).sort_values(ascending=True).plot(kind='barh', color='navy')
    plt.title('Top 10 REAL Factors Affecting Delivery Delay', fontsize=14)
    plt.xlabel('Relative Importance Score')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()
    # ۶. بصری‌سازی (ماتریس اغتشاش)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title('Confusion Matrix: Predicted vs Actual', fontsize=14)
    plt.xlabel('Predicted Label (0: On-Time, 1: Late)')
    plt.ylabel('Actual Label')
    plt.show()
# برای اجرای تابع، کافیست خط زیر را اجرا کنید:
run_supply_chain_analysis(df)
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
# ۱. پیش‌پردازش و آماده‌سازی (مشابه کد قبلی برای حفظ پایداری)
df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
df['order_month'] = df['order_date'].dt.month
df['order_day_of_week'] = df['order_date'].dt.dayofweek
df['order_hour'] = df['order_date'].dt.hour
# تعیین هدف: ریسک تأخیر در ارسال
y = df['Late_delivery_risk']
# لیست سیاه برای حذف تقلب و شناسه‌های بی‌ارزش
black_list = [
'is_late', 'Late_delivery_risk', 'Delivery Status', 'Order Status',
'Days for shipping (real)', 'delivery_gap', 'Order Customer Id',
'Order Item Cardprod Id', 'Customer Id', 'Order Id', 'Product Card Id',
'Customer Email', 'Customer Fname', 'Customer Lname', 'order_date',
'order date (DateOrders)', 'shipping date (DateOrders)'
]
# انتخاب ویژگی‌های عددی واقعی
X = df.select_dtypes(include=['number']).drop(columns=black_list, errors='ignore')
# ۲. تقسیم داده‌ها و آموزش مدل
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
# ۳. بخش XAI و SHAP برای بهینه‌سازی زنجیره تأمین
print("\n--- در حال محاسبه مقادیر SHAP برای تفسیر استراتژیک زنجیره تأمین ---")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# الف) نمودار خلاصه (Summary Plot) - برای شناسایی گلوگاه‌های اصلی تأخیر
plt.figure(figsize=(10, 6))
plt.title("Strategic Factors in Supply Chain Delay (SHAP Analysis)")
shap.summary_plot(shap_values, X_test)
# ب) نمودار وابستگی (Dependence Plot) - برای بهینه‌سازی یک ویژگی خاص
# مثلاً رابطه بین روزهای برنامه‌ریزی شده و ریسک واقعی تأخیر
top_feature = X.columns[np.argmax(model.feature_importances_)]
plt.figure(figsize=(8, 5))
shap.dependence_plot(top_feature, shap_values, X_test)
# ۴. خروجی نهایی دقت
from sklearn.metrics import accuracy_score