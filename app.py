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

# Global variable to store results
analysis_results = {}

def create_synthetic_target(df):
    """Create synthetic credit risk target variable"""
    df['risk_score'] = 0
    
    # Income factor (lower income = higher risk)
    income_threshold = df['income'].quantile(0.3)
    df['risk_score'] += (df['income'] < income_threshold).astype(int) * 2
    
    # House ownership factor (renting = higher risk)
    if 'house_ownership' in df.columns:
        df['risk_score'] += (df['house_ownership'] == 'rented').astype(int)
    
    # Car ownership factor (no car = higher risk)
    if 'car_ownership' in df.columns:
        df['risk_score'] += (df['car_ownership'] == 'no').astype(int)
    
    # Job stability factor (short job tenure = higher risk)
    if 'current_job_years' in df.columns:
        job_tenure_threshold = df['current_job_years'].quantile(0.3)
        df['risk_score'] += (df['current_job_years'] < job_tenure_threshold).astype(int)
    
    # Age factor (very young = higher risk)
    if 'age' in df.columns:
        df['risk_score'] += (df['age'] < 25).astype(int)
    
    # Create binary target (0 = low risk, 1 = high risk)
    risk_threshold = df['risk_score'].quantile(0.7)
    df['credit_risk'] = (df['risk_score'] >= risk_threshold).astype(int)
    
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    # Check for target column
    target_column = None
    possible_target_names = ['loan_status', 'status', 'target', 'label', 'class', 
                             'default', 'loan_default', 'Status', 'TARGET', 'risk', 
                             'credit_risk']
    
    for col in df.columns:
        if col in possible_target_names or 'status' in col.lower() or 'target' in col.lower():
            target_column = col
            break
    
    # Create synthetic target if none exists
    if target_column is None:
        df = create_synthetic_target(df)
        target_column = 'credit_risk'
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Separate features and target
    y = df[target_column].values
    feature_cols = [col for col in df.columns if col not in [target_column, 'risk_score']]
    X_raw = df[feature_cols].copy()
    
    # Identify numerical and categorical columns
    numerical_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle outliers in numerical columns
    for col in numerical_cols:
        if col in X_raw.columns:
            Q1 = X_raw[col].quantile(0.25)
            Q3 = X_raw[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            X_raw[col] = X_raw[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Encode categorical variables
    X_encoded = X_raw.copy()
    le_dict = {}
    for col in categorical_cols:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            le_dict[col] = le
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Get target distribution
    unique, counts = np.unique(y, return_counts=True)
    target_distribution = dict(zip(unique.tolist(), counts.tolist()))
    
    return X_scaled, y, target_column, target_distribution, len(feature_cols)

def perform_pca_analysis(X_scaled):
    """Perform PCA analysis"""
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

def train_models(X_scaled, y):
    """Train and evaluate multiple models"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis()
    }
    
    results = {}
    
    for model_name, model in models.items():
        accuracy = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        precision = cross_val_score(model, X_scaled, y, cv=cv, scoring='precision_macro')
        recall = cross_val_score(model, X_scaled, y, cv=cv, scoring='recall_macro')
        f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_macro')
        
        results[model_name] = {
            'accuracy': float(accuracy.mean()),
            'accuracy_std': float(accuracy.std()),
            'precision': float(precision.mean()),
            'precision_std': float(precision.std()),
            'recall': float(recall.mean()),
            'recall_std': float(recall.std()),
            'f1': float(f1.mean()),
            'f1_std': float(f1.std())
        }
    
    return results

def get_detailed_evaluation(X_scaled, y, best_model_name):
    """Get detailed evaluation on test set"""
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Select best model
    models = {
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis()
    }
    
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'], output_dict=True)
    
    return {
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_f1': float(f1_score(y_test, y_pred, average='macro')),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def generate_visualization(results, confusion_matrix_data):
    """Generate visualization plots and return as base64"""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Model Comparison - Accuracy
    ax1 = plt.subplot(2, 3, 1)
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    errors = [results[m]['accuracy_std'] for m in models]
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    ax1.bar(range(len(models)), accuracies, yerr=errors, capsize=5, 
            color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Model Comparison - Accuracy', fontweight='bold', fontsize=11)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Model Comparison - F1-Score
    ax2 = plt.subplot(2, 3, 2)
    f1_scores = [results[m]['f1'] for m in models]
    f1_errors = [results[m]['f1_std'] for m in models]
    ax2.bar(range(len(models)), f1_scores, yerr=f1_errors, capsize=5,
            color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('F1-Score', fontweight='bold')
    ax2.set_title('Model Comparison - F1-Score', fontweight='bold', fontsize=11)
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Precision vs Recall
    ax3 = plt.subplot(2, 3, 3)
    precisions = [results[m]['precision'] for m in models]
    recalls = [results[m]['recall'] for m in models]
    for i, model in enumerate(models):
        ax3.scatter(recalls[i], precisions[i], s=250, c=colors[i], 
                    alpha=0.7, edgecolors='black', linewidth=2, label=model)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax3.set_xlabel('Recall', fontweight='bold')
    ax3.set_ylabel('Precision', fontweight='bold')
    ax3.set_title('Precision vs Recall', fontweight='bold', fontsize=11)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. Confusion Matrix
    ax4 = plt.subplot(2, 3, 4)
    cm = np.array(confusion_matrix_data)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Low Risk', 'High Risk'], 
                yticklabels=['Low Risk', 'High Risk'])
    ax4.set_xlabel('Predicted', fontweight='bold')
    ax4.set_ylabel('Actual', fontweight='bold')
    ax4.set_title('Confusion Matrix', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Credit Risk Prediction System</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 {
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-align: center;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 40px;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 60px 40px;
                text-align: center;
                background: #f8f9ff;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background: #eef1ff;
                border-color: #764ba2;
            }
            .upload-icon {
                font-size: 4em;
                margin-bottom: 20px;
            }
            input[type="file"] {
                display: none;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 50px;
                font-size: 1.1em;
                cursor: pointer;
                transition: transform 0.2s;
                margin-top: 20px;
            }
            .btn:hover {
                transform: scale(1.05);
            }
            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: scale(1);
            }
            #results {
                display: none;
                margin-top: 40px;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                margin: 15px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric-item {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }
            .metric-label {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 5px;
            }
            .metric-value {
                color: #667eea;
                font-size: 2em;
                font-weight: bold;
            }
            .model-comparison {
                margin-top: 30px;
            }
            .model-item {
                background: #f8f9ff;
                padding: 20px;
                margin: 10px 0;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }
            .progress-bar {
                background: #e0e0e0;
                height: 10px;
                border-radius: 5px;
                margin-top: 10px;
                overflow: hidden;
            }
            .progress-fill {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                height: 100%;
                transition: width 0.5s;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 40px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .chart-container {
                margin-top: 30px;
                text-align: center;
            }
            .chart-container img {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Credit Risk Prediction System</h1>
            <p class="subtitle">Advanced Machine Learning for Credit Risk Assessment</p>
            
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
                    Analyze Dataset
                </button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 20px; color: #667eea; font-weight: bold;">
                    Processing your data... This may take a minute.
                </p>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            let selectedFile = null;
            
            document.getElementById('fileInput').addEventListener('change', function(e) {
                selectedFile = e.target.files[0];
                if (selectedFile) {
                    document.getElementById('fileName').textContent = '✓ ' + selectedFile.name;
                    document.getElementById('analyzeBtn').disabled = false;
                }
            });
            
            async function analyzeData() {
                if (!selectedFile) return;
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                document.getElementById('uploadArea').style.display = 'none';
                document.getElementById('loading').style.display = 'block';
                
                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    displayResults(data);
                } catch (error) {
                    alert('Error analyzing data: ' + error.message);
                    document.getElementById('uploadArea').style.display = 'block';
                    document.getElementById('loading').style.display = 'none';
                }
            }
            
            function displayResults(data) {
                document.getElementById('loading').style.display = 'none';
                const resultsDiv = document.getElementById('results');
                resultsDiv.style.display = 'block';
                
                let html = `
                    <div class="metric-card">
                        <h2>📊 Analysis Complete!</h2>
                        <p style="margin-top: 10px;">Dataset processed successfully</p>
                    </div>
                    
                    <div class="metric-grid">
                        <div class="metric-item">
                            <div class="metric-label">Dataset Size</div>
                            <div class="metric-value">${data.dataset_shape}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Features</div>
                            <div class="metric-value">${data.n_features}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Best Accuracy</div>
                            <div class="metric-value">${(data.best_model.accuracy * 100).toFixed(2)}%</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">F1-Score</div>
                            <div class="metric-value">${(data.best_model.f1 * 100).toFixed(2)}%</div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h2>🏆 Best Model: ${data.best_model.name}</h2>
                        <div class="metric-grid">
                            <div style="color: white;">
                                <strong>Accuracy:</strong> ${(data.best_model.accuracy * 100).toFixed(2)}%
                            </div>
                            <div style="color: white;">
                                <strong>Precision:</strong> ${(data.best_model.precision * 100).toFixed(2)}%
                            </div>
                            <div style="color: white;">
                                <strong>Recall:</strong> ${(data.best_model.recall * 100).toFixed(2)}%
                            </div>
                            <div style="color: white;">
                                <strong>F1-Score:</strong> ${(data.best_model.f1 * 100).toFixed(2)}%
                            </div>
                        </div>
                    </div>
                    
                    <div class="model-comparison">
                        <h2>📈 Model Comparison</h2>
                `;
                
                for (const [modelName, metrics] of Object.entries(data.models)) {
                    html += `
                        <div class="model-item">
                            <h3>${modelName}</h3>
                            <p>Accuracy: ${(metrics.accuracy * 100).toFixed(2)}% | 
                               F1-Score: ${(metrics.f1 * 100).toFixed(2)}%</p>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${metrics.accuracy * 100}%"></div>
                            </div>
                        </div>
                    `;
                }
                
                html += `
                    </div>
                    
                    <div class="chart-container">
                        <h2>📊 Visualization</h2>
                        <img src="data:image/png;base64,${data.visualization}" alt="Analysis Charts">
                    </div>
                    
                    <div class="metric-card" style="margin-top: 30px;">
                        <h2>💡 Key Findings</h2>
                        <ul style="list-style: none; padding: 20px;">
                            <li>✓ Dataset contains ${data.dataset_shape} records</li>
                            <li>✓ Low Risk: ${data.target_distribution[0]} records</li>
                            <li>✓ High Risk: ${data.target_distribution[1]} records</li>
                            <li>✓ PCA: ${data.pca.n_components_95} components for 95% variance</li>
                            <li>✓ Best performing model: ${data.best_model.name}</li>
                        </ul>
                    </div>
                `;
                
                resultsDiv.innerHTML = html;
            }
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
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Preprocess data
        X_scaled, y, target_column, target_distribution, n_features = preprocess_data(df)
        
        # PCA analysis
        pca_results = perform_pca_analysis(X_scaled)
        
        # Train models
        model_results = train_models(X_scaled, y)
        
        # Find best model
        best_model_name = max(model_results.items(), key=lambda x: x[1]['f1'])[0]
        best_model_metrics = model_results[best_model_name]
        
        # Detailed evaluation
        detailed_eval = get_detailed_evaluation(X_scaled, y, best_model_name)
        
        # Generate visualization
        visualization = generate_visualization(model_results, detailed_eval['confusion_matrix'])
        
        # Prepare response
        response = {
            'dataset_shape': f"{df.shape[0]:,} rows × {df.shape[1]} columns",
            'n_features': n_features,
            'target_column': target_column,
            'target_distribution': target_distribution,
            'pca': pca_results,
            'models': model_results,
            'best_model': {
                'name': best_model_name,
                'accuracy': best_model_metrics['accuracy'],
                'precision': best_model_metrics['precision'],
                'recall': best_model_metrics['recall'],
                'f1': best_model_metrics['f1']
            },
            'detailed_evaluation': detailed_eval,
            'visualization': visualization
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Credit Risk API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)