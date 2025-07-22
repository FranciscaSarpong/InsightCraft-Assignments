import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score

# Configure Streamlit app
st.set_page_config(page_title="Student Performance Analysis", layout="wide")

# Load dataset function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('C:/Users/Francesca Manu/PycharmProjects/Group_Work_Research_Methods/StudentsPerformance.csv')
        df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
        df['pass'] = np.where(df['average_score'] >= 60, 1, 0)
        return df
    except Exception as e:
        st.error(f"Error in load_data: {e}")
        return None

# Train logistic regression model
@st.cache_resource
def train_logistic_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    return model, scaler, X_train.columns, metrics, y_test, y_pred

# Train random forest regressor model
@st.cache_resource
def train_random_forest_model(X, y, n_estimators=100, max_depth=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include='object').columns),
            ('num', 'passthrough', X.select_dtypes(exclude='object').columns)
        ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    return model, preprocessor, X_train.columns, metrics, y_test, y_pred

# Sidebar for navigation and file upload
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "Logistic Regression"])
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (optional)", type=["csv"])

# Data loading with error handling
df = None
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Uploaded file did not result in a valid Pandas DataFrame")
        df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
        df['pass'] = np.where(df['average_score'] >= 60, 1, 0)
    else:
        df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Please upload a file or ensure 'StudentsPerformance.csv' is in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Ensure df is a valid DataFrame
if df is None or not isinstance(df, pd.DataFrame):
    st.error("No valid dataset loaded. Please upload a valid CSV file.")
    st.stop()

# EDA Section
if page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.write("This section provides a comprehensive analysis of the Students Performance dataset.")

    # 1. Dataset Overview
    st.header("1. Dataset Overview")
    if st.checkbox("Preview raw data"):
        st.dataframe(df.head())

    # 2. Summary Statistics
    st.header("2. Summary Statistics")
    try:
        st.write("**Numerical Columns**")
        st.dataframe(df.describe())
        st.write("**Categorical Columns**")
        st.dataframe(df.describe(include='object'))
    except AttributeError as e:
        st.error(f"Error accessing describe method: {e}. Please ensure the dataset is valid.")
        st.stop()

    # 3. Missing Values
    st.header("3. Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0] if missing.sum() > 0 else pd.Series({"No missing values": 0}))

    # 4. Distribution of Scores
    st.header("4. Distribution of Scores")
    score_type = st.selectbox("Select score type", ["math score", "reading score", "writing score", "average_score"])
    fig = px.histogram(df, x=score_type, nbins=30, color="gender", marginal="box",
                       title=f"{score_type.capitalize()} Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # 5. Box Plots by Category
    st.header("5. Performance by Categories")
    category = st.selectbox("Select category", ["gender", "race/ethnicity", "parental level of education", "lunch",
                                               "test preparation course"])
    fig2 = px.box(df, x=category, y=score_type, color="gender",
                  title=f"{score_type.capitalize()} by {category}")
    st.plotly_chart(fig2, use_container_width=True)

    # 6. Correlation Analysis
    st.header("6. Correlation Analysis")
    encoded_df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        encoded_df[col] = le.fit_transform(df[col])
    # Drop 'average_score' and 'pass' columns
    encoded_df = encoded_df.drop(columns=['average_score', 'pass'])
    corr_matrix = encoded_df.corr()
    fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
    st.plotly_chart(fig3, use_container_width=True)

    # 7. Grouped Statistics
    st.header("7. Grouped Statistics")
    for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
        grouped = df.groupby(col)['average_score'].mean().reset_index()
        st.subheader(f'Average Score by {col}')
        st.dataframe(grouped)
        fig = px.bar(grouped, x=col, y='average_score', title=f'Average Score by {col}')
        st.plotly_chart(fig, use_container_width=True)

    # 8. Download Cleaned Dataset
    csv = df.to_csv(index=False)
    st.download_button("Download Cleaned Dataset", csv, "cleaned_dataset.csv", "text/csv")

# Logistic Regression Section
elif page == "Logistic Regression":
    st.title("Logistic Regression Analysis")
    st.write("Predict whether a student passes (average score ≥ 60).")

    # Preprocess data
    features = st.multiselect("Select prediction features",
                              ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'],
                              default=['gender', 'parental level of education', 'lunch'])
    if not features:
        st.warning("Please select at least one feature to proceed.")
    else:
        X = pd.get_dummies(df[features], drop_first=True)
        y = df['pass']
        try:
            model, scaler, feature_names, metrics, y_test, y_pred = train_logistic_model(X, y)
        except Exception as e:
            st.error(f"Error training logistic regression model: {e}")
            st.stop()

        # Display Metrics
        st.header("1. Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        col2.metric("Precision", f"{metrics['precision']:.2f}")
        col3.metric("Recall", f"{metrics['recall']:.2f}")
        col4.metric("F1-Score", f"{metrics['f1']:.2f}")

        # Confusion Matrix
        st.header("2. Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

        # Feature Importance
        st.header("3. Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        })
        st.dataframe(feature_importance.sort_values(by='Coefficient', key=abs, ascending=False))
        fig = px.bar(feature_importance, x='Coefficient', y='Feature', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)
        csv_importance = feature_importance.to_csv(index=False)
        st.download_button("Download Feature Importance", csv_importance, "feature_importance.csv", "text/csv")

        # Prediction Form
        st.header("4. Make a Prediction")
        with st.form("prediction_form"):
            inputs = {}
            for feature in features:
                inputs[feature] = st.selectbox(f"{feature}", df[feature].unique())
            submitted = st.form_submit_button("Predict")
            if submitted:
                input_data = pd.DataFrame([inputs])
                input_data = pd.get_dummies(input_data)
                input_data = input_data.reindex(columns=feature_names, fill_value=0)
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                st.subheader("Prediction Result")
                st.write(f"**Outcome**: {'Pass' if prediction == 1 else 'Fail'}")
                st.write(f"**Probability of Passing**: {probability:.2%}")

