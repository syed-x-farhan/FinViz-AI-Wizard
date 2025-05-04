# FinViz AI Wizard

A powerful Streamlit web application for financial data analysis and machine learning. This application allows users to analyze financial data using various machine learning models and provides interactive visualizations of the results.

**Instructor**: Dr. Usama Arshad  
**Course**: Programming for Finance

## Features

- Upload financial datasets or fetch real-time stock data from Yahoo Finance
- Complete machine learning pipeline implementation
- Multiple model options:
  - Linear Regression
  - Logistic Regression
  - K-Means Clustering
- Interactive data preprocessing and feature engineering
- Comprehensive model evaluation metrics
- Beautiful visualizations using Plotly
- Dark mode interface with modern UI elements
- Download results as CSV

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd finviz-ai-wizard
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Follow the step-by-step workflow:
   - Welcome: Get started with the application
   - Data Input: Upload your dataset or fetch stock data
   - Preprocessing: Clean and prepare your data
   - Feature Engineering: Create new features and select important ones
   - Model Selection: Choose your machine learning model
   - Training: Train the model with your data
   - Evaluation: View model performance metrics
   - Visualization: Explore interactive visualizations of results

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- yfinance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
