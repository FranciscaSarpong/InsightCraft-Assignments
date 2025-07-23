import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ——— App Config ———
st.set_page_config(page_title="Student Performance Analysis", layout="wide")

# ——— Load Dataset ———
@st.cache_data
def load_data():
    path = 'C:/Users/Francesca Manu/PycharmProjects/Group_Work_Research_Methods/StudentsPerformance.csv'  # ← point to your local CSV in the project
    df = pd.read_csv(path)
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['pass'] = np.where(df['average_score'] >= 60, 1, 0)
    return df

df = load_data()

# ——— Sidebar Navigation ———
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "Logistic Regression"])

# ——— EDA Page ———
if page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.write("A deep dive into the Students Performance dataset.")

    # 1. Dataset Overview
    st.header("1. Dataset Overview")
    if st.checkbox("Preview raw data"):
        st.dataframe(df.head())

    # 2. Summary Statistics
    st.header("2. Summary Statistics")
    st.subheader("Numerical Columns")
    st.dataframe(df.describe())
    st.subheader("Categorical Columns")
    st.dataframe(df.describe(include='object'))

    # 3. Missing Values
    st.header("3. Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0] if missing.sum() > 0 else pd.DataFrame({"No missing": [0]}))

    # 4. Score Distributions
    st.header("4. Distribution of Scores")
    score_type = st.selectbox(
        "Select score type",
        ["math score", "reading score", "writing score", "average_score"]
    )
    fig = px.histogram(
        df, x=score_type, nbins=30, color="gender", marginal="box",
        title=f"{score_type.capitalize()} Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5. Performance by Category
    st.header("5. Performance by Categories")
    category = st.selectbox(
        "Select category",
        ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
    )
    fig2 = px.box(
        df, x=category, y=score_type, color="gender",
        title=f"{score_type.capitalize()} by {category}"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 6. Correlation Analysis
    st.header("6. Correlation Analysis")
    encoded = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        encoded[col] = le.fit_transform(df[col])
    corr = encoded.corr()
    fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
    st.plotly_chart(fig3, use_container_width=True)

    # 7. Download Cleaned Data
    csv = df.to_csv(index=False)
    st.download_button("Download Cleaned Dataset", csv, "cleaned_dataset.csv", "text/csv")


# ——— Logistic Regression Page ———
else:
    st.title("Logistic Regression Analysis")
    st.write("Predicting Pass/Fail based on background factors.")

    # Feature selection
    features = st.multiselect(
        "Select features",
        ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'],
        default=['gender', 'parental level of education', 'lunch']
    )
    if not features:
        st.warning("Please select at least one feature.")
        st.stop()

    # Prepare data
    X = pd.get_dummies(df[features], drop_first=True)
    y = df['pass']

    # Train-test split & scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Model training
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

    st.header("1. Model Performance")
    cols = st.columns(len(metrics))
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.2f}")

    # Confusion matrix
    st.header("2. Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig_cm)

    # Feature coefficients
    st.header("3. Feature Importance (Coefficients)")
    coefs = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)
    st.dataframe(coefs)
    fig_imp = px.bar(coefs, x='Coefficient', y='Feature', title='Feature Importance')
    st.plotly_chart(fig_imp, use_container_width=True)
