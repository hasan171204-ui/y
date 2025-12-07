import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, make_scorer
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("📊 General Machine Learning Dashboard")

# ======================================================
# SIDEBAR — UPLOAD
# ======================================================
st.sidebar.title("📁 Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.warning("Silakan upload file CSV terlebih dahulu.")
    st.stop()

data = pd.read_csv(uploaded)
st.sidebar.success("Data berhasil di-load!")

# Session State
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "train_cols" not in st.session_state:
    st.session_state.train_cols = None
if "encoders" not in st.session_state:
    st.session_state.encoders = {}  # simpan semua label encoder
if "cv_results" not in st.session_state:
    st.session_state.cv_results = None


# ======================================================
# TABS
# ======================================================
tab1, tab2 = st.tabs(["⚙ Modeling & Prediction", "📈 Visualization"])


# ======================================================
# TAB 1 : MODELING
# ======================================================
with tab1:

    st.header("⚙ Modeling & Prediction")

    # -----------------------------------------
    # PILIH TARGET & FEATURES
    # -----------------------------------------
    st.subheader("1. Pilih Target & Features")

    target = st.selectbox("Pilih Target Variable", data.columns)
    features = st.multiselect(
        "Pilih Feature Variables",
        [c for c in data.columns if c != target],
        default=[c for c in data.columns if c != target]
    )

    if len(features) == 0:
        st.error("Minimal 1 feature harus dipilih.")
        st.stop()

    df = data.copy()
    X = df[features].copy()
    y = df[target].copy()

    # -----------------------------------------
    # PREPROCESSING
    # -----------------------------------------
    st.subheader("2. Preprocessing & Feature Selection")

    num_cols = X.select_dtypes(include=['float', 'int']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Impute
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode()[0])

        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])

        st.session_state.encoders[c] = le  # simpan encoder

    # Feature Selection
    use_feature_selection = st.checkbox("Gunakan Feature Selection (SelectKBest)", value=True)
    
    if use_feature_selection:
        k_features = st.slider("Jumlah Features", min_value=1, max_value=X.shape[1], 
                               value=min(10, X.shape[1]))
        try:
            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_new = selector.fit_transform(X, y)
            selected_cols = X.columns[selector.get_support()]
            st.session_state.train_cols = selected_cols
            X = pd.DataFrame(X_new, columns=selected_cols)
            st.success(f"Selected Features: {list(selected_cols)}")
        except:
            st.info("SelectKBest dilewati.")
            st.session_state.train_cols = X.columns
    else:
        st.session_state.train_cols = X.columns

    # =======================================
    # MODEL SELECTION & CONFIGURATION
    # =======================================
    st.subheader("3. Pilih Model & Konfigurasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox("Model:", ["Logistic Regression", "Random Forest", "XGBoost"])
        
        if model_name == "Logistic Regression":
            base_model = LogisticRegression(max_iter=2000, random_state=42)
        elif model_name == "Random Forest":
            n_est = st.number_input("N Estimators", value=100, min_value=10, max_value=500, step=10)
            base_model = RandomForestClassifier(n_estimators=n_est, random_state=42)
        else:
            n_est = st.number_input("N Estimators", value=100, min_value=10, max_value=500, step=10)
            base_model = XGBClassifier(n_estimators=n_est, eval_metric='logloss', random_state=42)
    
    with col2:
        # SMOTE Configuration
        use_smote = st.checkbox("Gunakan SMOTE (untuk imbalanced data)", value=True)
        
        if use_smote and y.nunique() == 2:
            value_counts = y.value_counts(normalize=True)
            if value_counts.min() < 0.4:
                st.warning(f"⚠️ Data imbalanced: {y.value_counts().to_dict()}")
            else:
                st.info(f"ℹ️ Data distribution: {y.value_counts().to_dict()}")
        
        # K-Fold Configuration
        use_kfold = st.checkbox("Gunakan K-Fold Cross Validation", value=True)
        if use_kfold:
            n_splits = st.slider("Jumlah Folds (K)", min_value=2, max_value=10, value=5)

    # =======================================
    # CREATE PIPELINE
    # =======================================
    # Pipeline: Scaler -> SMOTE (optional) -> Model
    if use_smote and y.nunique() == 2 and y.value_counts(normalize=True).min() < 0.4:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', base_model)
        ])
        st.info("🔧 Pipeline: StandardScaler → SMOTE → Model")
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', base_model)
        ])
        st.info("🔧 Pipeline: StandardScaler → Model")

    # Train-Test Split
    test_size = st.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, 
        stratify=y if y.nunique() == 2 else None
    )

    # =======================================
    # TRAIN MODEL WITH K-FOLD CV
    # =======================================
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            
            # ===== K-FOLD CROSS VALIDATION =====
            if use_kfold:
                st.subheader("📊 K-Fold Cross Validation Results")
                
                # Define scoring metrics
                scoring = {
                    'accuracy': 'accuracy',
                    'precision': make_scorer(precision_score, zero_division=0),
                    'recall': make_scorer(recall_score, zero_division=0),
                    'f1': make_scorer(f1_score, zero_division=0)
                }
                
                if y.nunique() == 2:
                    scoring['roc_auc'] = 'roc_auc'
                
                # Perform cross-validation
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_results = cross_validate(
                    pipeline, X_train, y_train, 
                    cv=cv, 
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1
                )
                
                st.session_state.cv_results = cv_results
                
                # Display CV results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    acc_mean = cv_results['test_accuracy'].mean()
                    acc_std = cv_results['test_accuracy'].std()
                    st.metric("Accuracy", f"{acc_mean:.3f} ± {acc_std:.3f}")
                
                with col2:
                    prec_mean = cv_results['test_precision'].mean()
                    prec_std = cv_results['test_precision'].std()
                    st.metric("Precision", f"{prec_mean:.3f} ± {prec_std:.3f}")
                
                with col3:
                    rec_mean = cv_results['test_recall'].mean()
                    rec_std = cv_results['test_recall'].std()
                    st.metric("Recall", f"{rec_mean:.3f} ± {rec_std:.3f}")
                
                with col4:
                    f1_mean = cv_results['test_f1'].mean()
                    f1_std = cv_results['test_f1'].std()
                    st.metric("F1-Score", f"{f1_mean:.3f} ± {f1_std:.3f}")
                
                if y.nunique() == 2:
                    auc_mean = cv_results['test_roc_auc'].mean()
                    auc_std = cv_results['test_roc_auc'].std()
                    st.write(f"**AUC-ROC:** {auc_mean:.3f} ± {auc_std:.3f}")
                
                # Visualize CV scores
                st.subheader("📈 Cross-Validation Score Distribution")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot 1: Scores across folds
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
                fold_scores = pd.DataFrame({
                    metric: cv_results[f'test_{metric}'] 
                    for metric in metrics_to_plot
                })
                fold_scores.index = [f'Fold {i+1}' for i in range(n_splits)]
                
                fold_scores.plot(kind='bar', ax=axes[0], width=0.8)
                axes[0].set_title('Scores Across Folds')
                axes[0].set_ylabel('Score')
                axes[0].set_xlabel('Fold')
                axes[0].legend(title='Metrics')
                axes[0].set_ylim([0, 1])
                axes[0].grid(axis='y', alpha=0.3)
                plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Plot 2: Box plot
                fold_scores_long = fold_scores.reset_index().melt(
                    id_vars='index', 
                    var_name='Metric', 
                    value_name='Score'
                )
                sns.boxplot(data=fold_scores_long, x='Metric', y='Score', ax=axes[1], palette='Set2')
                axes[1].set_title('Score Distribution Across All Folds')
                axes[1].set_ylim([0, 1])
                axes[1].grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Detailed fold results table
                with st.expander("📋 Detailed Fold Results"):
                    detailed_results = pd.DataFrame({
                        'Fold': [f'Fold {i+1}' for i in range(n_splits)],
                        'Accuracy': cv_results['test_accuracy'],
                        'Precision': cv_results['test_precision'],
                        'Recall': cv_results['test_recall'],
                        'F1-Score': cv_results['test_f1']
                    })
                    if y.nunique() == 2:
                        detailed_results['ROC-AUC'] = cv_results['test_roc_auc']
                    
                    st.dataframe(detailed_results.style.format({
                        'Accuracy': '{:.4f}',
                        'Precision': '{:.4f}',
                        'Recall': '{:.4f}',
                        'F1-Score': '{:.4f}',
                        'ROC-AUC': '{:.4f}' if y.nunique() == 2 else None
                    }))
            
            # ===== FINAL MODEL TRAINING ON FULL TRAINING SET =====
            st.subheader("🎯 Final Model Evaluation on Test Set")
            
            # Train on full training set
            pipeline.fit(X_train, y_train)
            
            # Predict on test set
            preds = pipeline.predict(X_test)
            prob = pipeline.predict_proba(X_test)[:, 1] if y.nunique() == 2 else None
            
            # Save to session state
            st.session_state.model = pipeline
            st.session_state.scaler = pipeline.named_steps['scaler']
            
            # Display test set metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
            with col2:
                st.metric("Precision", f"{precision_score(y_test, preds, zero_division=0):.3f}")
            with col3:
                st.metric("Recall", f"{recall_score(y_test, preds, zero_division=0):.3f}")
            with col4:
                st.metric("F1-Score", f"{f1_score(y_test, preds, zero_division=0):.3f}")
            
            # Specificity
            try:
                cm = confusion_matrix(y_test, preds)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    st.write(f"**Specificity:** {spec:.3f}")
            except:
                pass
            
            if prob is not None:
                auc = roc_auc_score(y_test, prob)
                st.write(f"**AUC-ROC:** {auc:.3f}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", 
                           cmap="Blues", ax=ax, cbar=False)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            
            with col2:
                if prob is not None:
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, prob)
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    ax2.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
                    ax2.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
                    ax2.set_xlabel("False Positive Rate")
                    ax2.set_ylabel("True Positive Rate")
                    ax2.set_title("ROC Curve")
                    ax2.legend()
                    ax2.grid(alpha=0.3)
                    st.pyplot(fig2)
            
            # Feature Importance
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                st.subheader("📊 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': st.session_state.train_cols,
                    'Importance': pipeline.named_steps['classifier'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', 
                           ax=ax3, palette='viridis')
                ax3.set_title("Top 10 Feature Importance")
                st.pyplot(fig3)
            
            # Comparison: CV vs Test Set
            if use_kfold:
                st.subheader("📊 Comparison: Cross-Validation vs Test Set")
                
                comparison_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'CV Mean': [
                        cv_results['test_accuracy'].mean(),
                        cv_results['test_precision'].mean(),
                        cv_results['test_recall'].mean(),
                        cv_results['test_f1'].mean()
                    ],
                    'CV Std': [
                        cv_results['test_accuracy'].std(),
                        cv_results['test_precision'].std(),
                        cv_results['test_recall'].std(),
                        cv_results['test_f1'].std()
                    ],
                    'Test Set': [
                        accuracy_score(y_test, preds),
                        precision_score(y_test, preds, zero_division=0),
                        recall_score(y_test, preds, zero_division=0),
                        f1_score(y_test, preds, zero_division=0)
                    ]
                })
                
                st.dataframe(comparison_df.style.format({
                    'CV Mean': '{:.4f}',
                    'CV Std': '{:.4f}',
                    'Test Set': '{:.4f}'
                }))
                
                # Visualization
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                x = np.arange(len(comparison_df))
                width = 0.35
                
                ax4.bar(x - width/2, comparison_df['CV Mean'], width, 
                       label='CV Mean', yerr=comparison_df['CV Std'], capsize=5, alpha=0.8)
                ax4.bar(x + width/2, comparison_df['Test Set'], width, 
                       label='Test Set', alpha=0.8)
                
                ax4.set_xlabel('Metrics')
                ax4.set_ylabel('Score')
                ax4.set_title('Cross-Validation vs Test Set Performance')
                ax4.set_xticks(x)
                ax4.set_xticklabels(comparison_df['Metric'])
                ax4.legend()
                ax4.set_ylim([0, 1])
                ax4.grid(axis='y', alpha=0.3)
                
                st.pyplot(fig4)

    # =======================================
    # PREDICTION FORM
    # =======================================
    st.subheader("🧮 Prediction Form")

    if st.session_state.model is None:
        st.info("Train model dulu.")
    else:
        with st.form("prediction_form"):
            st.write("Masukkan data untuk prediksi:")
            
            input_data = {}
            
            cols = st.columns(3)
            col_idx = 0
            
            for col in features:
                with cols[col_idx % 3]:
                    if col in num_cols:
                        median_val = float(df[col].median())
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        input_data[col] = st.number_input(
                            col, 
                            value=median_val,
                            min_value=min_val,
                            max_value=max_val
                        )
                    else:
                        options = sorted(df[col].unique().tolist())
                        input_data[col] = st.selectbox(col, options)
                col_idx += 1
            
            submitted = st.form_submit_button("🔮 Predict", type="primary")
            
            if submitted:
                input_df = pd.DataFrame([input_data])

                # Preprocessing consistency
                for c in num_cols:
                    if c in input_df.columns:
                        input_df[c] = input_df[c].fillna(df[c].median())

                for c in cat_cols:
                    if c in input_df.columns:
                        enc = st.session_state.encoders[c]
                        try:
                            input_df[c] = enc.transform(input_df[c].astype(str))
                        except:
                            st.error(f"Unknown category in {c}")
                            st.stop()

                # Feature selection alignment
                input_df = input_df[st.session_state.train_cols]

                # Predict using pipeline
                pred = st.session_state.model.predict(input_df)[0]
                
                # Probability
                if hasattr(st.session_state.model, 'predict_proba'):
                    proba = st.session_state.model.predict_proba(input_df)[0]

                # Display result
                st.success("### 🔎 Prediction Result")
                
                # Mapping label
                if y.nunique() == 2:
                    classes = sorted(y.unique())
                    mapping = {classes[0]: "Tidak Mengalami Stroke", classes[1]: "Mengalami Stroke"}
                    pred_label = mapping[pred]
                    st.write(f"## **{pred_label}**")
                    
                    if hasattr(st.session_state.model, 'predict_proba'):
                        st.write("**Probability:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(mapping[classes[0]], f"{proba[0]*100:.2f}%")
                        with col2:
                            st.metric(mapping[classes[1]], f"{proba[1]*100:.2f}%")
                else:
                    st.write(f"## **Predicted {target}: {pred}**")


# ======================================================
# TAB 2 : VISUALIZATION
# ======================================================
with tab2:

    st.header("📈 Visualization")

    plot_type = st.selectbox("Jenis Plot:", [
        "Histogram", "Boxplot", "Scatterplot", "Pairplot",
        "Bar Chart", "Count Plot", "Pie Chart", "Crosstab Heatmap",
        "Correlation Heatmap"
    ])
    
    numeric_cols = data.select_dtypes(include=['float', 'int']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # ===== VISUALISASI NUMERIK =====
    if plot_type == "Histogram":
        if len(numeric_cols) == 0:
            st.warning("Tidak ada kolom numerik dalam dataset")
        else:
            col = st.selectbox("Kolom", numeric_cols)
            bins = st.slider("Number of bins", 10, 100, 30)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[col].dropna(), kde=True, bins=bins, ax=ax)
            ax.set_title(f"Distribusi {col}")
            st.pyplot(fig)

    elif plot_type == "Boxplot":
        if len(numeric_cols) == 0:
            st.warning("Tidak ada kolom numerik dalam dataset")
        else:
            col = st.selectbox("Kolom", numeric_cols)
            by_col = st.selectbox("Group by (opsional)", ["None"] + categorical_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if by_col != "None":
                sns.boxplot(data=data, y=col, x=by_col, ax=ax)
                plt.xticks(rotation=45, ha='right')
            else:
                sns.boxplot(y=data[col], ax=ax)
            ax.set_title(f"Boxplot {col}")
            st.pyplot(fig)

    elif plot_type == "Scatterplot":
        if len(numeric_cols) < 2:
            st.warning("Butuh minimal 2 kolom numerik")
        else:
            x = st.selectbox("X-axis", numeric_cols)
            y = st.selectbox("Y-axis", numeric_cols)
            hue_col = st.selectbox("Hue (opsional)", ["None"] + categorical_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_col != "None":
                sns.scatterplot(data=data, x=x, y=y, hue=hue_col, ax=ax)
            else:
                sns.scatterplot(data=data, x=x, y=y, ax=ax)
            ax.set_title(f"Scatterplot: {x} vs {y}")
            st.pyplot(fig)

    elif plot_type == "Pairplot":
        if len(numeric_cols) == 0:
            st.warning("Tidak ada kolom numerik")
        else:
            selected = st.multiselect("Kolom", numeric_cols, 
                                     default=numeric_cols[:min(3, len(numeric_cols))] if len(numeric_cols) >= 3 else numeric_cols)
            if len(selected) > 0:
                hue_col = st.selectbox("Hue (opsional)", ["None"] + categorical_cols)
                if hue_col != "None":
                    fig = sns.pairplot(data[selected + [hue_col]], hue=hue_col)
                else:
                    fig = sns.pairplot(data[selected])
                st.pyplot(fig)
            else:
                st.warning("Pilih minimal 1 kolom")

    elif plot_type == "Correlation Heatmap":
        if len(numeric_cols) == 0:
            st.warning("Tidak ada kolom numerik")
        else:
            corr = data[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                       center=0, square=True, ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

    # ===== VISUALISASI KATEGORIK =====
    elif plot_type == "Bar Chart":
        if len(categorical_cols) == 0:
            st.warning("Tidak ada kolom kategorik dalam dataset")
        else:
            col = st.selectbox("Kolom", categorical_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts = data[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_title(f"Frekuensi {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

    elif plot_type == "Count Plot":
        if len(categorical_cols) == 0:
            st.warning("Tidak ada kolom kategorik dalam dataset")
        else:
            col = st.selectbox("Kolom", categorical_cols)
            hue_col = st.selectbox("Group by (opsional)", ["None"] + categorical_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_col != "None" and hue_col != col:
                sns.countplot(data=data, x=col, hue=hue_col, ax=ax)
            else:
                sns.countplot(data=data, x=col, ax=ax)
            ax.set_title(f"Count Plot {col}")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

    elif plot_type == "Pie Chart":
        if len(categorical_cols) == 0:
            st.warning("Tidak ada kolom kategorik dalam dataset")
        else:
            col = st.selectbox("Kolom", categorical_cols)
            fig, ax = plt.subplots(figsize=(10, 8))
            value_counts = data[col].value_counts()
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title(f"Distribusi {col}")
            st.pyplot(fig)

    elif plot_type == "Crosstab Heatmap":
        if len(categorical_cols) < 2:
            st.warning("Butuh minimal 2 kolom kategorik")
        else:
            col1 = st.selectbox("Kolom 1", categorical_cols)
            col2 = st.selectbox("Kolom 2", [c for c in categorical_cols if c != col1])
            fig, ax = plt.subplots(figsize=(10, 8))
            crosstab = pd.crosstab(data[col1], data[col2])
            sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
            ax.set_title(f"Crosstab: {col1} vs {col2}")
            st.pyplot(fig)