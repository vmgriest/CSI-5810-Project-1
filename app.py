from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

N_FOLDS = 5  # You can change this to any value (3, 5, 10, etc.)

analysis_results = {}

def create_synthetic_target(df):
    df['risk_score'] = 0
    
    income_threshold = df['income'].quantile(0.3)
    df['risk_score'] += (df['income'] < income_threshold).astype(int) * 2
    
    if 'house_ownership' in df.columns:
        df['risk_score'] += (df['house_ownership'] == 'rented').astype(int)
    
    if 'car_ownership' in df.columns:
        df['risk_score'] += (df['car_ownership'] == 'no').astype(int)
    
    if 'current_job_years' in df.columns:
        job_tenure_threshold = df['current_job_years'].quantile(0.3)
        df['risk_score'] += (df['current_job_years'] < job_tenure_threshold).astype(int)
    
    if 'age' in df.columns:
        df['risk_score'] += (df['age'] < 25).astype(int)
    
    risk_threshold = df['risk_score'].quantile(0.7)
    df['credit_risk'] = (df['risk_score'] >= risk_threshold).astype(int)
    
    return df

def preprocess_data(df):
    target_column = None
    possible_target_names = ['loan_status', 'status', 'target', 'label', 'class', 
                             'default', 'loan_default', 'Status', 'TARGET', 'risk', 
                             'credit_risk']
    
    for col in df.columns:
        if col in possible_target_names or 'status' in col.lower() or 'target' in col.lower():
            target_column = col
            break
    
    if target_column is None:
        df = create_synthetic_target(df)
        target_column = 'credit_risk'
    
    df = df.drop_duplicates()
    
    df = df.fillna(df.median(numeric_only=True))
    
    y = df[target_column].values
    feature_cols = [col for col in df.columns if col not in [target_column, 'risk_score']]
    X_raw = df[feature_cols].copy()
    
    numerical_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in numerical_cols:
        if col in X_raw.columns:
            Q1 = X_raw[col].quantile(0.25)
            Q3 = X_raw[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            X_raw[col] = X_raw[col].clip(lower=lower_bound, upper=upper_bound)
    
    X_encoded = X_raw.copy()
    le_dict = {}
    for col in categorical_cols:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            le_dict[col] = le
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    unique, counts = np.unique(y, return_counts=True)
    target_distribution = dict(zip(unique.tolist(), counts.tolist()))
    
    return X_scaled, y, target_column, target_distribution, len(feature_cols)

def perform_pca_analysis(X_scaled):
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    explained_var_cumsum = np.cumsum(explained_var)
    
    n_components_95 = np.argmax(explained_var_cumsum >= 0.95) + 1
    
    return {
        'n_components_95': int(n_components_95),
        'cumulative_variance': float(explained_var_cumsum[n_components_95-1]),
        'explained_variance': explained_var[:10].tolist()
    }

def train_models(X_scaled, y, n_folds=N_FOLDS):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    models = {
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Calculate scores for each fold
        accuracy_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='precision_macro')
        recall_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='recall_macro')
        f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_macro')
        
        results[model_name] = {
            'accuracy_mean': float(accuracy_scores.mean()),
            'accuracy_std': float(accuracy_scores.std()),
            'accuracy_scores': accuracy_scores.tolist(),
            'precision_mean': float(precision_scores.mean()),
            'precision_std': float(precision_scores.std()),
            'precision_scores': precision_scores.tolist(),
            'recall_mean': float(recall_scores.mean()),
            'recall_std': float(recall_scores.std()),
            'recall_scores': recall_scores.tolist(),
            'f1_mean': float(f1_scores.mean()),
            'f1_std': float(f1_scores.std()),
            'f1_scores': f1_scores.tolist(),
            'n_folds': n_folds
        }
        
        print(f"  Accuracy:  {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f}")
        print(f"  Precision: {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
        print(f"  Recall:    {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
        print(f"  F1-Score:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}\n")
    
    return results

def get_detailed_evaluation(X_scaled, y, best_model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis()
    }
    
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'], output_dict=True)
    
    return {
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_f1': float(f1_score(y_test, y_pred, average='macro')),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def generate_visualization(results, confusion_matrix_data, n_folds):
    fig = plt.figure(figsize=(18, 12))
    
    models = list(results.keys())
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    
    ax1 = plt.subplot(2, 3, 1)
    accuracies = [results[m]['accuracy_mean'] for m in models]
    errors = [results[m]['accuracy_std'] for m in models]
    bars = ax1.bar(range(len(models)), accuracies, yerr=errors, capsize=5, 
            color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
    ax1.set_title(f'{n_folds}-Fold CV: Model Accuracy (Mean ± Std)', fontweight='bold', fontsize=12)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Baseline (50%)')
    
    for i, (bar, acc, err) in enumerate(zip(bars, accuracies, errors)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
                f'{acc:.3f}\n±{err:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2 = plt.subplot(2, 3, 2)
    f1_scores = [results[m]['f1_mean'] for m in models]
    f1_errors = [results[m]['f1_std'] for m in models]
    bars2 = ax2.bar(range(len(models)), f1_scores, yerr=f1_errors, capsize=5,
            color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('F1-Score', fontweight='bold', fontsize=11)
    ax2.set_title(f'{n_folds}-Fold CV: F1-Score (Mean ± Std)', fontweight='bold', fontsize=12)
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, f1, err) in enumerate(zip(bars2, f1_scores, f1_errors)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
                f'{f1:.3f}\n±{err:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3 = plt.subplot(2, 3, 3)
    accuracy_data = [results[m]['accuracy_scores'] for m in models]
    bp = ax3.boxplot(accuracy_data, labels=models, patch_artist=True, 
                     showmeans=True, meanline=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax3.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
    ax3.set_title(f'Accuracy Distribution Across {n_folds} Folds', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim([0, 1])
    
    ax4 = plt.subplot(2, 3, 4)
    precisions = [results[m]['precision_mean'] for m in models]
    recalls = [results[m]['recall_mean'] for m in models]
    precision_errs = [results[m]['precision_std'] for m in models]
    recall_errs = [results[m]['recall_std'] for m in models]
    
    for i, model in enumerate(models):
        ax4.errorbar(recalls[i], precisions[i], 
                    xerr=recall_errs[i], yerr=precision_errs[i],
                    fmt='o', markersize=15, capsize=5, capthick=2,
                    color=colors[i], alpha=0.7, linewidth=2,
                    markeredgecolor='black', markeredgewidth=2, label=model)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Perfect Balance')
    ax4.set_xlabel('Recall (Mean ± Std)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Precision (Mean ± Std)', fontweight='bold', fontsize=11)
    ax4.set_title('Precision vs Recall Trade-off', fontweight='bold', fontsize=12)
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    ax5 = plt.subplot(2, 3, 5)
    cm = np.array(confusion_matrix_data)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, 
                cbar_kws={'label': 'Count'}, 
                xticklabels=['Low Risk', 'High Risk'], 
                yticklabels=['Low Risk', 'High Risk'],
                linewidths=1, linecolor='black')
    ax5.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax5.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax5.set_title('Confusion Matrix (Test Set)', fontweight='bold', fontsize=12)
    
    ax6 = plt.subplot(2, 3, 6)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        if metric == 'Accuracy':
            values = [results[m]['accuracy_mean'] for m in models]
        elif metric == 'Precision':
            values = [results[m]['precision_mean'] for m in models]
        elif metric == 'Recall':
            values = [results[m]['recall_mean'] for m in models]
        else:
            values = [results[m]['f1_mean'] for m in models]
        
        ax6.bar(x + i*width - 0.3, values, width, label=metric, alpha=0.8)
    
    ax6.set_xlabel('Models', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax6.set_title(f'{n_folds}-Fold CV: All Metrics Comparison', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax6.legend(fontsize=9)
    ax6.set_ylim([0, 1])
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

@app.route('/')
def index():
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Credit Risk Prediction System</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }}
            h1 {{
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-align: center;
            }}
            .subtitle {{
                text-align: center;
                color: #666;
                margin-bottom: 20px;
                font-size: 1.1em;
            }}
            .cv-info {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
                font-size: 1.2em;
                font-weight: bold;
            }}
            .upload-area {{
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 60px 40px;
                text-align: center;
                background: #f8f9ff;
                transition: all 0.3s;
            }}
            .upload-area:hover {{
                background: #eef1ff;
                border-color: #764ba2;
            }}
            .upload-icon {{
                font-size: 4em;
                margin-bottom: 20px;
            }}
            input[type="file"] {{
                display: none;
            }}
            .btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 50px;
                font-size: 1.1em;
                cursor: pointer;
                transition: transform 0.2s;
                margin-top: 20px;
            }}
            .btn:hover {{
                transform: scale(1.05);
            }}
            .btn:disabled {{
                background: #ccc;
                cursor: not-allowed;
                transform: scale(1);
            }}
            #results {{
                display: none;
                margin-top: 40px;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                margin: 15px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-item {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9em;
                margin-bottom: 5px;
            }}
            .metric-value {{
                color: #667eea;
                font-size: 2em;
                font-weight: bold;
            }}
            .metric-std {{
                color: #999;
                font-size: 0.9em;
                margin-top: 5px;
            }}
            .model-comparison {{
                margin-top: 30px;
            }}
            .model-item {{
                background: #f8f9ff;
                padding: 25px;
                margin: 15px 0;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }}
            .model-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            .badge {{
                background: #4caf50;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
            }}
            .metrics-row {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }}
            .metric-box {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .metric-box .label {{
                color: #666;
                font-size: 0.85em;
                margin-bottom: 5px;
            }}
            .metric-box .value {{
                color: #667eea;
                font-size: 1.3em;
                font-weight: bold;
            }}
            .metric-box .std {{
                color: #999;
                font-size: 0.8em;
                margin-top: 3px;
            }}
            .progress-bar {{
                background: #e0e0e0;
                height: 10px;
                border-radius: 5px;
                margin-top: 10px;
                overflow: hidden;
            }}
            .progress-fill {{
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                height: 100%;
                transition: width 0.5s;
            }}
            .loading {{
                display: none;
                text-align: center;
                padding: 40px;
            }}
            .spinner {{
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .chart-container {{
                margin-top: 30px;
                text-align: center;
                background: #f8f9ff;
                padding: 30px;
                border-radius: 15px;
            }}
            .chart-container img {{
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .fold-info {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .fold-info strong {{
                color: #856404;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Credit Risk Prediction System</h1>
            <p class="subtitle">Advanced Machine Learning with {N_FOLDS}-Fold Cross-Validation</p>
            
            <div class="cv-info">
                📊 Using {N_FOLDS}-Fold Stratified Cross-Validation<br>
                <span style="font-size: 0.85em; opacity: 0.9;">Mean ± Standard Deviation reported for all metrics</span>
            </div>
            
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <h2>Upload Your Dataset</h2>
                <p style="color: #666; margin: 20px 0;">
                    Drag and drop your CSV file here or click to browse
                </p>
                <input type="file" id="fileInput" accept=".csv">
                <label for="fileInput" class="btn">Choose File</label>
                <p id="fileName" style="margin-top: 20px; color: #667eea; font-weight: bold;"></p>
                <button class="btn" id="analyzeBtn" onclick="analyzeData()" disabled>
                    Analyze Dataset with {N_FOLDS}-Fold CV
                </button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 20px; color: #667eea; font-weight: bold;">
                    Performing {N_FOLDS}-fold cross-validation... This may take a minute.
                </p>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            let selectedFile = null;
            
            document.getElementById('fileInput').addEventListener('change', function(e) {{
                selectedFile = e.target.files[0];
                if (selectedFile) {{
                    document.getElementById('fileName').textContent = '✓ ' + selectedFile.name;
                    document.getElementById('analyzeBtn').disabled = false;
                }}
            }});
            
            async function analyzeData() {{
                if (!selectedFile) return;
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                document.getElementById('uploadArea').style.display = 'none';
                document.getElementById('loading').style.display = 'block';
                
                try {{
                    const response = await fetch('/api/analyze', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const data = await response.json();
                    displayResults(data);
                }} catch (error) {{
                    alert('Error analyzing data: ' + error.message);
                    document.getElementById('uploadArea').style.display = 'block';
                    document.getElementById('loading').style.display = 'none';
                }}
            }}
            
            function displayResults(data) {{
                document.getElementById('loading').style.display = 'none';
                const resultsDiv = document.getElementById('results');
                resultsDiv.style.display = 'block';
                
                let html = `
                    <div class="metric-card">
                        <h2>📊 Analysis Complete with ${{data.n_folds}}-Fold Cross-Validation!</h2>
                        <p style="margin-top: 10px;">Dataset processed successfully with robust validation</p>
                    </div>
                    
                    <div class="fold-info">
                        <strong>Cross-Validation Details:</strong> Each model was trained and evaluated ${{data.n_folds}} times 
                        on different data splits. The reported metrics show the mean performance ± standard deviation across all folds.
                    </div>
                    
                    <div class="metric-grid">
                        <div class="metric-item">
                            <div class="metric-label">Best Accuracy</div>
                            <div class="metric-value">${{(data.best_model.accuracy_mean * 100).toFixed(2)}}%</div>
                            <div class="metric-std">± ${{(data.best_model.accuracy_std * 100).toFixed(2)}}%</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Best F1-Score</div>
                            <div class="metric-value">${{(data.best_model.f1_mean * 100).toFixed(2)}}%</div>
                            <div class="metric-std">± ${{(data.best_model.f1_std * 100).toFixed(2)}}%</div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h2>🏆 Best Model: ${{data.best_model.name}}</h2>
                        <p style="margin: 10px 0; opacity: 0.9;">Selected based on highest F1-Score across ${{data.n_folds}} folds</p>
                        <div class="metrics-row">
                            <div class="metric-box">
                                <div class="label">Accuracy</div>
                                <div class="value">${{(data.best_model.accuracy_mean * 100).toFixed(2)}}%</div>
                                <div class="std">± ${{(data.best_model.accuracy_std * 100).toFixed(2)}}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="label">Precision</div>
                                <div class="value">${{(data.best_model.precision_mean * 100).toFixed(2)}}%</div>
                                <div class="std">± ${{(data.best_model.precision_std * 100).toFixed(2)}}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="label">Recall</div>
                                <div class="value">${{(data.best_model.recall_mean * 100).toFixed(2)}}%</div>
                                <div class="std">± ${{(data.best_model.recall_std * 100).toFixed(2)}}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="label">F1-Score</div>
                                <div class="value">${{(data.best_model.f1_mean * 100).toFixed(2)}}%</div>
                                <div class="std">± ${{(data.best_model.f1_std * 100).toFixed(2)}}%</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="model-comparison">
                        <h2>📈 Model Comparison (${{data.n_folds}}-Fold Cross-Validation)</h2>
                        <p style="color: #666; margin: 10px 0;">All metrics show mean ± standard deviation across folds</p>
                `;
                
                for (const [modelName, metrics] of Object.entries(data.models)) {{
                    const isBest = modelName === data.best_model.name;
                    html += `
                        <div class="model-item">
                            <div class="model-header">
                                <h3>${{modelName}}</h3>
                                ${{isBest ? '<span class="badge">Best Model</span>' : ''}}
                            </div>
                            
                            <div class="metrics-row">
                                <div class="metric-box">
                                    <div class="label">Accuracy</div>
                                    <div class="value">${{(metrics.accuracy_mean * 100).toFixed(2)}}%</div>
                                    <div class="std">± ${{(metrics.accuracy_std * 100).toFixed(2)}}%</div>
                                </div>
                                <div class="metric-box">
                                    <div class="label">Precision</div>
                                    <div class="value">${{(metrics.precision_mean * 100).toFixed(2)}}%</div>
                                    <div class="std">± ${{(metrics.precision_std * 100).toFixed(2)}}%</div>
                                </div>
                                <div class="metric-box">
                                    <div class="label">Recall</div>
                                    <div class="value">${{(metrics.recall_mean * 100).toFixed(2)}}%</div>
                                    <div class="std">± ${{(metrics.recall_std * 100).toFixed(2)}}%</div>
                                </div>
                                <div class="metric-box">
                                    <div class="label">F1-Score</div>
                                    <div class="value">${{(metrics.f1_mean * 100).toFixed(2)}}%</div>
                                    <div class="std">± ${{(metrics.f1_std * 100).toFixed(2)}}%</div>
                                </div>
                            </div>
                            
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${{metrics.accuracy_mean * 100}}%"></div>
                            </div>
                        </div>
                    `;
                }}
                
                html += `
                    </div>
                    
                    <div class="chart-container">
                        <h2>📊 Comprehensive Visualization</h2>
                        <p style="color: #666; margin: 10px 0 20px 0;">
                            Visual analysis of ${{data.n_folds}}-fold cross-validation results including error bars and distribution plots
                        </p>
                        <img src="data:image/png;base64,${{data.visualization}}" alt="Analysis Charts">
                    </div>
                    
                    <div class="metric-card" style="margin-top: 30px;">
                        <h2>💡 Key Findings</h2>
                        <ul style="list-style: none; padding: 20px; line-height: 2;">
                            <li>✓ Dataset: ${{data.dataset_shape}}</li>
                            <li>✓ Low Risk: ${{data.target_distribution[0].toLocaleString()}} records (${{((data.target_distribution[0] / (data.target_distribution[0] + data.target_distribution[1])) * 100).toFixed(1)}}%)</li>
                            <li>✓ High Risk: ${{data.target_distribution[1].toLocaleString()}} records (${{((data.target_distribution[1] / (data.target_distribution[0] + data.target_distribution[1])) * 100).toFixed(1)}}%)</li>
                            <li>✓ Cross-Validation: ${{data.n_folds}} folds with stratified sampling</li>
                            <li>✓ Best Model: ${{data.best_model.name}} (F1: ${{(data.best_model.f1_mean * 100).toFixed(2)}}% ± ${{(data.best_model.f1_std * 100).toFixed(2)}}%)</li>
                            <li>✓ PCA Components for 95% variance: ${{data.pca.n_components_95}}</li>
                            <li>✓ All metrics validated with mean ± std across folds</li>
                        </ul>
                    </div>
                    
                    <div style="margin-top: 30px; padding: 20px; background: #e8f5e9; border-radius: 10px; border-left: 4px solid #4caf50;">
                        <h3 style="color: #2e7d32; margin-bottom: 10px;">✅ Validation Complete</h3>
                        <p style="color: #1b5e20;">
                            Your model has been rigorously validated using ${{data.n_folds}}-fold stratified cross-validation. 
                            The reported mean and standard deviation provide confidence intervals for model performance. 
                            Lower standard deviation indicates more stable and reliable predictions.
                        </p>
                    </div>
                `;
                
                resultsDiv.innerHTML = html;
            }}
        </script>
    </body>
    </html>
    '''

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        df = pd.read_csv(file)
        
        print(f"Processing dataset: {file.filename}")
        print(f"Shape: {df.shape}")
        
        X_scaled, y, target_column, target_distribution, n_features = preprocess_data(df)
        
        pca_results = perform_pca_analysis(X_scaled)
        
        model_results = train_models(X_scaled, y, n_folds=N_FOLDS)
        
        best_model_name = max(model_results.items(), key=lambda x: x[1]['f1_mean'])[0]
        best_model_metrics = model_results[best_model_name]
        
        print(f"BEST MODEL: {best_model_name}")
        print(f"F1-Score: {best_model_metrics['f1_mean']:.4f} ± {best_model_metrics['f1_std']:.4f}")

        detailed_eval = get_detailed_evaluation(X_scaled, y, best_model_name)
        
        visualization = generate_visualization(model_results, detailed_eval['confusion_matrix'], N_FOLDS)
  
        response = {
            'dataset_shape': f"{df.shape[0]:,} rows × {df.shape[1]} columns",
            'n_features': n_features,
            'n_folds': N_FOLDS,
            'target_column': target_column,
            'target_distribution': target_distribution,
            'pca': pca_results,
            'models': model_results,
            'best_model': {
                'name': best_model_name,
                'accuracy_mean': best_model_metrics['accuracy_mean'],
                'accuracy_std': best_model_metrics['accuracy_std'],
                'precision_mean': best_model_metrics['precision_mean'],
                'precision_std': best_model_metrics['precision_std'],
                'recall_mean': best_model_metrics['recall_mean'],
                'recall_std': best_model_metrics['recall_std'],
                'f1_mean': best_model_metrics['f1_mean'],
                'f1_std': best_model_metrics['f1_std']
            },
            'detailed_evaluation': detailed_eval,
            'visualization': visualization
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'message': 'Credit Risk API is running',
        'cross_validation_folds': N_FOLDS
    })

if __name__ == '__main__':
    print(f"Credit Risk Prediction System Starting...")
    print(f"Configuration: {N_FOLDS}-Fold Cross-Validation")
    print(f"Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)