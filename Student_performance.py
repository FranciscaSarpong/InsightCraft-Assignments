
# --- IMPORT SECTION ---
# Standard data science and visualization libraries
import streamlit as st  # For building the web interface
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations
import pickle  # For serializing Python objects
import os  # For file system operations
from PIL import Image  # For image handling

# Scikit-learn components for machine learning pipeline
from sklearn.linear_model import Ridge  # Regularized linear regression model
from sklearn.model_selection import cross_val_score, cross_validate  # Model evaluation
from sklearn.metrics import mean_squared_error, r2_score  # Performance metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Feature preprocessing
from sklearn.impute import SimpleImputer  # Handling missing values
from sklearn.pipeline import Pipeline  # ML pipeline construction
from sklearn.compose import ColumnTransformer  # Column-wise transformations
from sklearn.feature_selection import SequentialFeatureSelector  # Feature selection

# --- CONFIGURATION ---
DATA_DIR = "saved_data"   # Directory for storing processed data and artifacts
os.makedirs(DATA_DIR, exist_ok=True)  # Create directory if it doesn't exist

# --- HELPER FUNCTIONS ---
def save_artifact(obj, filename):
    """Serialize and save Python objects to disk for later use.

        Args:
            obj: Any Python object to be serialized
            filename: Name of the file to save to (within DATA_DIR)
        """

    with open(f"{DATA_DIR}/{filename}", 'wb') as f:
        pickle.dump(obj, f)


def load_artifact(filename):
    """Load previously saved Python objects from disk.

        Args:
            filename: Name of the file to load (within DATA_DIR)
        Returns:
            The deserialized Python object
        """
    with open(f"{DATA_DIR}/{filename}", 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("Loan_Default.csv")
    # Drop unnecessary columns
    df = df.drop(columns=['ID', 'dtir1', 'submission_of_application', 'year'], errors='ignore')
    # Save raw data for reference
    df.to_csv(f"{DATA_DIR}/1_raw_data.csv", index=False)
    return df

def create_preprocessor():
    """Create and fit a preprocessing pipeline for the loan data.
        """
    df = load_data()
    X = df.drop('loan_amount', axis=1)

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes( include=['object']).columns.tolist()
    save_artifact({'numerical': numerical_cols, 'categorical': categorical_cols},
                  "2_column_types.pkl")

    # Numerical feature pipeline: impute missing values with median and standardize
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Categorical feature pipeline: impute missing values with mode and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine pipelines in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])
    preprocessor.fit(X)
    save_artifact(preprocessor, "3_preprocessor.pkl")

    # Transform data and save processed version
    X_processed = preprocessor.transform(X)
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    all_features = np.concatenate([num_features, cat_features])
    processed_df = pd.DataFrame(X_processed, columns=all_features)
    processed_df['loan_amount'] = df['loan_amount'].values
    processed_df.to_csv(f"{DATA_DIR}/4_processed_data.csv", index=False)
    return preprocessor


# --- STREAMLIT PAGES ---
def Home_Page():
    """Render the home page with project overview and team information."""

    # LOGO
    logo = Image.open("Loan_Default_Image.png")

    st.image(logo, caption="", width=300)
    st.title("Loan Default Prediction Web App")
    st.markdown("""
        ---

        ### Project Overview

        This interactive web application was developed as part of an applied regression and machine learning course project. It simulates a real-world scenario where a data science team is tasked with building a system to *predict the probability of loan default* based on demographic and financial features.

        Users can explore the dataset, follow the full machine learning workflow, and interact with the final model to generate real-time predictions.

        ---


        ### Instructions
        Use the sidebar to navigate through the project steps:

        1. Data Import and Overview – Explore the dataset. 
        2. Data Preprocessing – Clean, impute, encode, and standardize
        3. Feature Selection – Best subset based on Ridge
        4. Model Training – Fit Ridge regression to selected features
        5. Model Evaluation – RMSE, R², and k-Fold CV
        6. Prediction – Enter values and predict default probability
        7. Conclusion – Insights and limitations

        ---

        *Developed by:* [Group 5 ]  
        *Tools:* Python, Streamlit, Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn, Pillow & Pickle.

        """)


    # Members section
    st.markdown("### Team Members")
    team_members = [
        ["1","Kingsley Sarfo", "22252461", "Project Coordinator","https://loan-predictor-hbbz24vwfzaue2qx4hwcat.streamlit.app"],
        ["2","Francisca Sarpong", "22255796", "Data Preprocessing","https://kftalde5ypwd5a3qqejuvo.streamlit.app"],
        ["3","George Owell", "22256146", "Model Evaluation","https://loandefaultpredictionapp-utmbic9znd7uzqqhs9zgo6.streamlit.app"],
        ["4","Barima Addo", "22254055", "UI Testing","https://loandefaultapp-ky4yy9kmt6ehsq8jqdcgs2.streamlit.app"],
        ["5","Marcus Akrobettoe", "11410687", "Feature Selection","https://models-loan-default-prediction.streamlit.app"]
    ]

    df = pd.DataFrame(team_members,
                      columns=["SN", "Name", "Student ID", "Role", "Deployment Link"])

    # Display as interactive table
    st.dataframe(df,
                 hide_index=True,
                 use_container_width=True,
                 column_config={
                     "Deployment Link": st.column_config.LinkColumn()
                 })

    # Project Overview Section
    st.markdown("""
        ###  Dataset Information:
        - Source: [Kaggle - Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
        - Target variable: loan_amount
        """)

def Data_Import_and_Overview_page():
    """Page for data upload, exploration and visualization."""
    st.title("1. Data Import and Overview")

    # File uploader widget
    st.subheader("Upload Your Loan Data (CSV)")
    uploaded_file = st.file_uploader("Upload your loan data CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop(columns=['ID', 'dtir1', 'submission_of_application', 'year'], errors='ignore')
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['df'] = df
        st.success("File uploaded and loaded! Data will be used throughout the app.")
    elif 'df' in st.session_state:
        df = st.session_state['df']
    else:
        st.warning("Please upload your data to use the app.")
        return

    st.dataframe(df.head())

    # Data summary statistics
    st.subheader("Summary Statistics")

    # Basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        if 'loan_amount' in df.columns:
            st.metric("Average Loan Amount", f"{df['loan_amount'].mean():,.2f}")

    # Numerical features summary
    st.markdown("**Numerical Features Summary**")
    st.dataframe(df.describe().T.style.format("{:.2f}"))

    # Categorical features summary
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        st.markdown("**Categorical Features Summary**")
        cat_summary = pd.DataFrame({
            'Unique Values': df[cat_cols].nunique(),
            'Most Common': df[cat_cols].mode().iloc[0],
            'Missing Values': df[cat_cols].isnull().sum()
        })
        st.dataframe(cat_summary)

    # Missing values analysis
    st.markdown("**Missing Values Analysis**")
    missing = df.isnull().sum().to_frame('Missing Values')
    missing['Percentage'] = (missing['Missing Values'] / len(df)) * 100
    st.dataframe(missing.style.format({'Percentage': '{:.2f}%'}))

    # ========================
    # 2. Data Visualizations
    # ========================
    st.subheader("Data Visualizations")

    # Target distribution (loan amount)
    if 'loan_amount' in df.columns:
        st.markdown("**Loan Amount Distribution**")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        sns.histplot(x='loan_amount', data=df, kde=True, ax=ax[0])
        ax[0].set_title('Loan Amount Distribution')
        ax[0].set_xlabel('Loan Amount')

        # Boxplot
        sns.boxplot(x='loan_amount', data=df, ax=ax[1])
        ax[1].set_title('Loan Amount Spread')
        ax[1].set_xlabel('Loan Amount')
        st.pyplot(fig)

    # Numerical distributions
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        st.markdown("**Numerical Features Distribution**")
        selected_num = st.multiselect("Select numerical features to visualize",
                                      num_cols,
                                      default=[col for col in ['income', 'Credit_Score', 'property_value'] if
                                               col in num_cols])

        if selected_num:
            fig, ax = plt.subplots(len(selected_num), 2, figsize=(14, 5 * len(selected_num)))
            for i, col in enumerate(selected_num):
                # Histogram
                sns.histplot(df[col], kde=True, ax=ax[i, 0])
                ax[i, 0].set_title(f'{col} Distribution')
                ax[i, 0].tick_params(axis='x', rotation=45)

                # Boxplot
                sns.boxplot(x=df[col], ax=ax[i, 1])
                ax[i, 1].set_title(f'{col} Boxplot')
                ax[i, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            st.pyplot(fig)

            # Scatterplots of numerical features vs loan amount
            if 'loan_amount' in df.columns:
                st.markdown("**Relationships with Loan Amount**")

                fig, ax = plt.subplots(1, 2, figsize=(14, 6))

                # 1. Income vs Loan Amount
                if 'income' in df.columns:
                    sns.scatterplot(x='income', y='loan_amount', data=df, ax=ax[0])
                    ax[0].set_title('Loan Amount vs Income')
                    ax[0].set_xlabel('Income ()')
                    ax[0].set_ylabel('Loan Amount')
                else:
                    sns.scatterplot(x=selected_num[0], y='loan_amount', data=df, ax=ax[0])
                    ax[0].set_title(f'Loan Amount vs {selected_num[0]}')
                    ax[0].set_xlabel(selected_num[0])
                    ax[0].set_ylabel('Loan Amount')

                # 2. Credit Score vs Loan Amount
                if 'Credit_Score' in df.columns:
                    sns.scatterplot(x='Credit_Score', y='loan_amount', data=df, ax=ax[1])
                    ax[1].set_title('Loan Amount vs Credit Score')
                    ax[1].set_xlabel('Credit Score')
                    ax[1].set_ylabel('Loan Amount')
                elif len(selected_num) > 1:
                    sns.scatterplot(x=selected_num[1], y='loan_amount', data=df, ax=ax[1])
                    ax[1].set_title(f'Loan Amount vs {selected_num[1]}')
                    ax[1].set_xlabel(selected_num[1])
                    ax[1].set_ylabel('Loan Amount')

                plt.tight_layout()
                st.pyplot(fig)

    # Correlation matrix
    if len(num_cols) > 1 and 'loan_amount' in num_cols:
        st.markdown("**Correlation Matrix (with Loan Amount)**")
        corr_matrix = df[num_cols].corr()

        # Highlight correlations with loan amount
        loan_corrs = corr_matrix['loan_amount'].sort_values(key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    center=0, ax=ax)
        ax.set_title("Feature Correlations with Loan Amount")
        st.pyplot(fig)

        # Top correlations with loan amount
        st.markdown("**Top Correlations with Loan Amount**")
        st.dataframe(loan_corrs.to_frame('Correlation').iloc[1:11])  # Exclude self-correlation

    # Categorical visualizations vs loan amount
    if len(cat_cols) > 0 and 'loan_amount' in df.columns:
        st.markdown("**Categorical Features vs Loan Amount**")
        selected_cat = st.selectbox("Select categorical feature", cat_cols)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Countplot
        sns.countplot(y=selected_cat, data=df, ax=ax[0],
                      order=df[selected_cat].value_counts().index)
        ax[0].set_title(f'{selected_cat} Distribution')

        # Boxplot of loan amount by category
        sns.boxplot(y=selected_cat, x='loan_amount', data=df,
                    ax=ax[1], order=df[selected_cat].value_counts().index)
        ax[1].set_title(f'Loan Amount by {selected_cat}')
        ax[1].set_xlabel('Loan Amount')

        st.pyplot(fig)

def Data_Preprocessing_page():
    """Manages the data preprocessing workflow:
            - Executes the preprocessing pipeline
            - Displays sample processed data"""

    st.title("2. Data Preprocessing")
    if st.button("Run Data Preprocessing"):
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        st.subheader("Processed Data Sample")
        st.dataframe(processed_df.head())
        st.success("Preprocessing completed and saved!")

def Feature_Selection_page():
    """Page for feature selection using sequential forward selection."""
    st.title("3. Feature Selection")
    try:
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        X = processed_df.drop('loan_amount', axis=1)
        y = processed_df['loan_amount']
    except:
        st.warning("Please complete preprocessing first")
        return

    # Feature selection interface
    st.subheader("Best Subset Selection (Ridge Regression)")
    max_features = st.slider("Maximum features to evaluate", 1, 15, 5)
    scoring_metric = st.selectbox("Selection metric", ['neg_root_mean_squared_error', 'r2'])

    if st.button("Run Best Subset Selection"):
        with st.spinner("Searching for optimal feature combinations..."):
            estimator = Ridge()
            sfs = SequentialFeatureSelector(estimator,
                                            n_features_to_select=max_features,
                                            direction='forward',
                                            scoring=scoring_metric,
                                            cv=5)
            sfs.fit(X, y)
            selected_features = X.columns[sfs.get_support()].tolist()
            save_artifact({
                'selected_features': selected_features,
                'selection_metric': scoring_metric,
                'support_mask': sfs.get_support()
            }, "5_best_subset_features.pkl")
            st.success(f"Selected {len(selected_features)} optimal features:")
            st.write(selected_features)
            st.subheader("Model Performance with Selected Features")
            cv_results = cross_validate(estimator,
                                       X[selected_features], y,
                                       cv=5,
                                       scoring=['neg_root_mean_squared_error', 'r2'])
            metrics_df = pd.DataFrame({
                'Mean': [abs(cv_results['test_neg_root_mean_squared_error'].mean()), cv_results['test_r2'].mean()],
                'Std': [cv_results['test_neg_root_mean_squared_error'].std(), cv_results['test_r2'].std()]
            }, index=['RMSE', 'R2 Score'])
            st.dataframe(metrics_df.style.format("{:.4f}"))
            estimator.fit(X[selected_features], y)
            if hasattr(estimator, 'coef_'):
                importance = pd.Series(np.abs(estimator.coef_),
                                       index=selected_features).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                importance.plot(kind='barh', ax=ax)
                ax.set_title("Feature Importance (Absolute Coefficients)")
                st.pyplot(fig)


def Model_Selection_And_Training_page():
    """Page for Ridge Regression model selection, cross-validation, and final training.

    This function handles:
    - Loading preprocessed data
    - Setting model hyperparameters
    - Running cross-validation
    - Training and saving the final model
    """

    # Page title
    st.title("4. Model Selection and Training")

    # Data loading with error handling
    try:
        # Load processed data from previous step
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        # Separate features (X) and target (y)
        X = processed_df.drop('loan_amount', axis=1)
        y = processed_df['loan_amount']
    except:
        # User-friendly error message if preprocessing isn't complete
        st.warning("Please complete preprocessing first")
        return  # Exit the function early

    # --- Model Parameter Configuration ---
    st.subheader("Ridge Regression Parameters")

    # Interactive alpha parameter slider
    # Default value of 1.0 is a reasonable starting point for Ridge
    alpha = st.slider("Ridge regularization strength (alpha)",
                      0.01,  # Minimum value
                      10.0,  # Maximum value
                      1.0)  # Default value

    # Initialize Ridge Regression model with selected alpha
    model = Ridge(alpha=alpha)

    # --- Cross-Validation Section ---
    if st.button("Run Cross-Validation"):
        # Perform 5-fold cross-validation
        # Using negative RMSE (sklearn convention - lower is better)
        rmse_scores = cross_val_score(model, X, y,
                                      cv=5,
                                      scoring='neg_root_mean_squared_error')

        # Also calculate R² scores
        r2_scores = cross_val_score(model, X, y,
                                    cv=5,
                                    scoring='r2')

        # Display performance metrics with standard deviation
        st.write(f"Mean RMSE: {abs(rmse_scores.mean()):.2f} (±{rmse_scores.std():.2f})")
        st.write(f"Mean R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")

        # Save cross-validation results for later use
        save_artifact({
            'rmse_scores': rmse_scores,
            'r2_scores': r2_scores,
            'params': model.get_params()  # Store model configuration
        }, "6_cv_results.pkl")

    # --- Final Model Training ---
    if st.button("Train Final Model"):
        # Fit model on entire dataset
        model.fit(X, y)

        # Save the trained model for later use
        save_artifact(model, "7_trained_model.pkl")

        # Generate and store predictions
        y_pred = model.predict(X)
        processed_df['Predicted_Amount'] = y_pred
        processed_df.to_csv(f"{DATA_DIR}/8_predictions.csv", index=False)

        # Store model and features in session state for other pages
        st.session_state['model'] = model
        st.session_state['features'] = X.columns.tolist()

        # Success notification
        st.success("Model trained and saved!")


def Model_Evaluation_page():
    """
    Displays comprehensive evaluation metrics and visualizations for the trained Ridge Regression model.

    This page:
    1. Loads the trained model and predictions
    2. Calculates key performance metrics (RMSE, R²)
    3. Shows actual vs predicted values plot
    4. Visualizes feature importance
    """

    st.title("5. Model Evaluation")

    # Load model artifacts with error handling
    try:
        # Load the serialized model
        model = load_artifact("7_trained_model.pkl")
        # Load the saved predictions
        predictions_df = pd.read_csv(f"{DATA_DIR}/8_predictions.csv")
    except:
        # User-friendly message if model hasn't been trained
        st.warning("Please train the model first")
        return  # Exit function early if artifacts missing

    # Extract true and predicted values
    y_true = predictions_df['loan_amount']  # Actual loan amounts
    y_pred = predictions_df['Predicted_Amount']  # Model predictions

    # --- Performance Metrics Section ---
    st.subheader("Model Performance")

    # Calculate Root Mean Squared Error (lower is better)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Calculate R-squared score (0-1, higher is better)
    r2 = r2_score(y_true, y_pred)

    # Display metrics in clean format
    st.metric("RMSE", f"{rmse:.2f}")  # Formatted to 2 decimal places
    st.metric("R²", f"{r2:.4f}")  # Formatted to 4 decimal places

    # --- Actual vs Predicted Visualization ---
    st.subheader("Actual vs Predicted Plot")

    # Create scatter plot of predictions
    fig, ax = plt.subplots()
    # Scatter plot with transparency
    ax.scatter(y_true, y_pred, alpha=0.3)
    # Add perfect prediction line (y=x)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            'r--')  # Red dashed line
    # Axis labels
    ax.set_xlabel("Actual Loan Amount")
    ax.set_ylabel("Predicted Loan Amount")
    # Display plot in Streamlit
    st.pyplot(fig)

    # --- Feature Importance Analysis ---
    st.subheader("Feature Importance")

    # Check if model has coefficients (Ridge regression does)
    if hasattr(model, 'coef_'):
        # Create absolute value of coefficients for importance
        importance = pd.Series(
            np.abs(model.coef_),
            index=predictions_df.drop(
                ['loan_amount', 'Predicted_Amount'],
                axis=1).columns
        )

        # Get top 5 most important features
        top_features = importance.nlargest(5)

        # Create horizontal bar plot
        fig, ax = plt.subplots()
        top_features.plot(kind='barh', ax=ax)

        # Display plot
        st.pyplot(fig)


def Interactive_Prediction_page():
    """
    Interactive page for making real-time loan default predictions using the trained model.

    This page:
    1. Loads required artifacts (model, preprocessor, feature list)
    2. Provides input form for user to enter applicant data
    3. Processes inputs and generates predictions
    4. Displays prediction results with formatting
    """

    st.title("6. Interactive Prediction")

    # --- Load Required Artifacts with Error Handling ---
    try:
        # Load trained Ridge Regression model
        model = load_artifact("7_trained_model.pkl")
        # Load preprocessor for feature transformation
        preprocessor = load_artifact("3_preprocessor.pkl")
        # Load original feature names and types
        original_features = load_artifact("2_column_types.pkl")
    except:
        # User-friendly message if prerequisites not met
        st.warning("Please complete model training first")
        return  # Exit function if artifacts not available

    # --- Input Form Section ---
    st.subheader("Enter Applicant Information")
    input_data = {}  # Dictionary to store user inputs

    # Create two columns for better form layout
    col1, col2 = st.columns(2)

    # Numerical Features Input
    with col1:
        st.markdown("**Numerical Features**")
        for feature in original_features['numerical']:
            # Create number input for each numerical feature
            input_data[feature] = st.number_input(
                label=f"{feature}",
                value=0.0,  # Default value
                step=0.01,  # Increment step
                format="%.2f"  # Display format
            )

    # Categorical Features Input
    with col2:
        st.markdown("**Categorical Features**")
        for feature in original_features['categorical']:
            # Create text input for each categorical feature
            input_data[feature] = st.text_input(
                label=f"{feature}",
                value="",  # Empty default
                help=f"Enter {feature} value"
            )

    # --- Prediction Section ---
    if st.button("Predict Default Amount", type="primary"):
        # Create DataFrame from user inputs (single row)
        df_template = pd.DataFrame([input_data])

        try:
            # Preprocess inputs using saved pipeline
            X_processed = preprocessor.transform(df_template)

            # Generate prediction
            predicted_amount = model.predict(X_processed)[0]  # Get single prediction

            # Display formatted result
            st.success(
                f"Predicted Loan Default Amount: ${predicted_amount:,.2f}",
            )

            # Optional: Show confidence interval or prediction range
            # st.info(f"Estimated range: ${predicted_amount*0.9:,.2f} - ${predicted_amount*1.1:,.2f}")

        except Exception as e:
            # Handle prediction errors gracefully
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check your input values and try again")


def Results_Interpretation_And_Conclusion_page():
    st.title("7. Results Interpretation and Conclusion")
    # Load the model and predictions
    try:
        model = load_artifact("7_trained_model.pkl")
        predictions_df = pd.read_csv(f"{DATA_DIR}/8_predictions.csv")
    except:
        st.warning("Please train the model first")
        return

    y_true = predictions_df['loan_amount']
    y_pred = predictions_df['Predicted_Amount']

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rmse_value = f"{rmse:.2f}"
    r2_value = f"{r2:.4f}"

    if hasattr(model, 'coef_'):
        importance = pd.Series(
            np.abs(model.coef_),
            index=predictions_df.drop(
                ['loan_amount', 'Predicted_Amount'],
                axis=1).columns
        )
        top_features = importance.nlargest(5)
        top_features_list = [f"{feat} ({imp:.2f})" for feat, imp in top_features.items()]
        top_features_str = ", ".join(top_features_list)
    else:
        top_features_str = "N/A"

    interpretation_md = f"""
    ## Model Performance Insights

    - The final Ridge Regression model achieved an RMSE of *{rmse_value}*, meaning that on average, predictions deviate from actual defaults by this amount.
    - The R² score of *{r2_value}* indicates that the model explains approximately *{float(r2)*100:.1f}%* of the variation in loan default amounts.

    ## Feature Insights

    - The most important features for prediction were: {top_features_str}

    ## Practical Impact

    - This model can help banks identify high-risk loans, personalize loan limits, and automate risk assessment workflows.
    - Outlier predictions (where model error is high) may reveal cases needing manual review.

    ## Limitations

    - The model’s accuracy depends on the quality and representativeness of the training data.
    - It may not fully account for macroeconomic shifts, fraud, or abrupt life events impacting borrowers.

    ## Future Work

    - Explore ensemble models (e.g XGBoost) for potentially higher accuracy.
    - Enhance interpretability using tools like SHAP.
    - Update the model periodically to capture changing economic conditions and borrower behaviors.
    """

    st.markdown(interpretation_md)
pages = {
    "Home Page": Home_Page,
    "Data Import and Overview": Data_Import_and_Overview_page,
    "Data Preprocessing": Data_Preprocessing_page,
    "Feature Selection": Feature_Selection_page,
    "Model Selection and Training": Model_Selection_And_Training_page,
    "Model Evaluation": Model_Evaluation_page,
    "Interactive Prediction": Interactive_Prediction_page,
    "Result Interpretation and Conclusion": Results_Interpretation_And_Conclusion_page,
}

selection = st.sidebar.selectbox("Select Page", list(pages.keys()))
pages[selection]()
