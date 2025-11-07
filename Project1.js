import React, { useState } from 'react';
import { Upload, BarChart3, TrendingUp, AlertCircle, CheckCircle, FileText, Database, Brain } from 'lucide-react';

const CreditRiskApp = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setResults({
        datasetShape: '252,000 rows × 11 columns',
        features: 11,
        lowRisk: 176400,
        highRisk: 75600,
        bestModel: 'Logistic Regression',
        accuracy: 0.8945,
        f1Score: 0.8834,
        precision: 0.8912,
        recall: 0.8756,
        pcaComponents: 7,
        models: [
          { name: 'k-NN (k=5)', accuracy: 0.8234, f1: 0.8123 },
          { name: 'Logistic Regression', accuracy: 0.8945, f1: 0.8834 },
          { name: 'Naive Bayes', accuracy: 0.8556, f1: 0.8445 },
          { name: 'LDA', accuracy: 0.8889, f1: 0.8778 }
        ]
      });
      setLoading(false);
      setActiveTab('results');
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-2">
            <Brain className="w-10 h-10 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-800">Credit Risk Prediction System</h1>
          </div>
          <p className="text-gray-600">Advanced Machine Learning for Credit Risk Assessment</p>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-lg mb-6">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex items-center gap-2 px-6 py-4 font-semibold transition-colors ${
                activeTab === 'upload'
                  ? 'border-b-2 border-indigo-600 text-indigo-600'
                  : 'text-gray-600 hover:text-indigo-600'
              }`}
            >
              <Upload className="w-5 h-5" />
              Upload Data
            </button>
            <button
              onClick={() => setActiveTab('results')}
              className={`flex items-center gap-2 px-6 py-4 font-semibold transition-colors ${
                activeTab === 'results'
                  ? 'border-b-2 border-indigo-600 text-indigo-600'
                  : 'text-gray-600 hover:text-indigo-600'
              }`}
              disabled={!results}
            >
              <BarChart3 className="w-5 h-5" />
              Results
            </button>
            <button
              onClick={() => setActiveTab('about')}
              className={`flex items-center gap-2 px-6 py-4 font-semibold transition-colors ${
                activeTab === 'about'
                  ? 'border-b-2 border-indigo-600 text-indigo-600'
                  : 'text-gray-600 hover:text-indigo-600'
              }`}
            >
              <FileText className="w-5 h-5" />
              About
            </button>
          </div>
        </div>

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Upload Dataset</h2>
            
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-indigo-500 transition-colors">
              <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-700 mb-2">
                Drop your CSV file here or click to browse
              </h3>
              <p className="text-gray-500 mb-4">
                Accepted format: CSV with credit data features
              </p>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="inline-block bg-indigo-600 text-white px-6 py-3 rounded-lg cursor-pointer hover:bg-indigo-700 transition-colors"
              >
                Select File
              </label>
              {file && (
                <div className="mt-4 text-green-600 font-semibold">
                  ✓ {file.name}
                </div>
              )}
            </div>

            {file && (
              <button
                onClick={handleUpload}
                disabled={loading}
                className="mt-6 w-full bg-green-600 text-white py-4 rounded-lg font-semibold hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Analyze Dataset'}
              </button>
            )}

            <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <Database className="w-8 h-8 text-blue-600 mb-2" />
                <h4 className="font-semibold text-gray-800">Data Processing</h4>
                <p className="text-sm text-gray-600">Automated cleaning and feature engineering</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <Brain className="w-8 h-8 text-green-600 mb-2" />
                <h4 className="font-semibold text-gray-800">ML Models</h4>
                <p className="text-sm text-gray-600">4 algorithms with cross-validation</p>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <TrendingUp className="w-8 h-8 text-purple-600 mb-2" />
                <h4 className="font-semibold text-gray-800">Insights</h4>
                <p className="text-sm text-gray-600">Comprehensive performance metrics</p>
              </div>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && results && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="text-sm text-gray-600 mb-1">Dataset Size</div>
                <div className="text-2xl font-bold text-indigo-600">{results.datasetShape}</div>
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="text-sm text-gray-600 mb-1">Best Accuracy</div>
                <div className="text-2xl font-bold text-green-600">{(results.accuracy * 100).toFixed(2)}%</div>
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="text-sm text-gray-600 mb-1">F1-Score</div>
                <div className="text-2xl font-bold text-blue-600">{(results.f1Score * 100).toFixed(2)}%</div>
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="text-sm text-gray-600 mb-1">PCA Components</div>
                <div className="text-2xl font-bold text-purple-600">{results.pcaComponents}</div>
              </div>
            </div>

            {/* Best Model */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle className="w-6 h-6 text-green-600" />
                <h3 className="text-xl font-bold text-gray-800">Best Model: {results.bestModel}</h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-sm text-gray-600">Accuracy</div>
                  <div className="text-lg font-semibold text-gray-800">{(results.accuracy * 100).toFixed(2)}%</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Precision</div>
                  <div className="text-lg font-semibold text-gray-800">{(results.precision * 100).toFixed(2)}%</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Recall</div>
                  <div className="text-lg font-semibold text-gray-800">{(results.recall * 100).toFixed(2)}%</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">F1-Score</div>
                  <div className="text-lg font-semibold text-gray-800">{(results.f1Score * 100).toFixed(2)}%</div>
                </div>
              </div>
            </div>

            {/* Model Comparison */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Model Comparison</h3>
              <div className="space-y-3">
                {results.models.map((model, idx) => (
                  <div key={idx} className="border rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-semibold text-gray-800">{model.name}</span>
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                        model.name === results.bestModel
                          ? 'bg-green-100 text-green-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {model.name === results.bestModel ? 'Best' : 'Good'}
                      </span>
                    </div>
                    <div className="flex gap-6">
                      <div>
                        <span className="text-sm text-gray-600">Accuracy: </span>
                        <span className="font-semibold">{(model.accuracy * 100).toFixed(2)}%</span>
                      </div>
                      <div>
                        <span className="text-sm text-gray-600">F1-Score: </span>
                        <span className="font-semibold">{(model.f1 * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                    <div className="mt-2 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-indigo-600 rounded-full h-2"
                        style={{ width: `${model.accuracy * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Class Distribution */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Credit Risk Distribution</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">Low Risk</div>
                  <div className="text-2xl font-bold text-green-600">
                    {results.lowRisk.toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-500">
                    ({((results.lowRisk / (results.lowRisk + results.highRisk)) * 100).toFixed(1)}%)
                  </div>
                </div>
                <div className="bg-red-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">High Risk</div>
                  <div className="text-2xl font-bold text-red-600">
                    {results.highRisk.toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-500">
                    ({((results.highRisk / (results.lowRisk + results.highRisk)) * 100).toFixed(1)}%)
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* About Tab */}
        {activeTab === 'about' && (
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">About This System</h2>
            
            <div className="prose max-w-none">
              <h3 className="text-xl font-semibold text-gray-800 mb-3">Overview</h3>
              <p className="text-gray-600 mb-4">
                This Credit Risk Prediction System uses advanced machine learning algorithms to assess credit risk
                based on various financial and demographic features. The system processes data, performs feature
                engineering, and trains multiple models to provide accurate risk predictions.
              </p>

              <h3 className="text-xl font-semibold text-gray-800 mb-3">Machine Learning Models</h3>
              <ul className="list-disc list-inside text-gray-600 mb-4 space-y-2">
                <li><strong>k-Nearest Neighbors (k-NN):</strong> Instance-based learning algorithm</li>
                <li><strong>Logistic Regression:</strong> Linear model for binary classification</li>
                <li><strong>Gaussian Naive Bayes:</strong> Probabilistic classifier based on Bayes' theorem</li>
                <li><strong>Linear Discriminant Analysis (LDA):</strong> Fisher's linear discriminant method</li>
              </ul>

              <h3 className="text-xl font-semibold text-gray-800 mb-3">Features</h3>
              <ul className="list-disc list-inside text-gray-600 mb-4 space-y-2">
                <li>Automated data preprocessing and cleaning</li>
                <li>Outlier detection and handling</li>
                <li>Feature scaling and encoding</li>
                <li>Principal Component Analysis (PCA) for dimensionality reduction</li>
                <li>5-fold stratified cross-validation</li>
                <li>Comprehensive performance metrics</li>
              </ul>

              <h3 className="text-xl font-semibold text-gray-800 mb-3">Evaluation Metrics</h3>
              <p className="text-gray-600 mb-4">
                The system evaluates models using multiple metrics including accuracy, precision, recall, and F1-score
                to ensure balanced performance across both risk classes.
              </p>

              <div className="bg-blue-50 border-l-4 border-blue-600 p-4 mt-6">
                <div className="flex items-start gap-2">
                  <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                  <div>
                    <h4 className="font-semibold text-blue-900">Note</h4>
                    <p className="text-blue-800 text-sm">
                      This system creates synthetic risk labels based on income, house ownership, car ownership,
                      job tenure, and age factors for demonstration purposes.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CreditRiskApp;