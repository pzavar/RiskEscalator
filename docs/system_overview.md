
# System Overview

## Architecture

The Risk Detection System is a Streamlit-based application that analyzes engineering team communications to identify and escalate potential risks. The system consists of several key components:

1. **Web Interface (`app.py`)**: Provides file upload, analysis controls, and results visualization
2. **Risk Detection (`risk_detection.py`)**: Core algorithms for identifying risks in conversations
3. **Message Analysis (`message_analysis.py`)**: Extracts insights from conversation patterns
4. **Data Visualization (`data_visualization.py`)**: Creates visual representations of the analysis
5. **Utilities (`utils.py`)**: Helper functions for data processing and report formatting

## Workflow

1. User uploads a CSV file containing Slack conversation data
2. The system validates the file format and displays a preview
3. When the user clicks "Analyze Conversation," the system:
   - Processes the conversation data
   - Detects potential risks using NLP techniques
   - Generates visualizations and insights
4. Results are displayed in three tabs:
   - Escalation Report: Detailed report of flagged messages
   - Visualizations: Charts and graphs showing risk patterns
   - Insights: Key findings and recommendations

## Key Features

- **Sentiment Analysis**: Uses NLTK's SentimentIntensityAnalyzer to evaluate message tone
- **Risk Keyword Detection**: Identifies technical terms that might indicate problems
- **Dismissive Language Detection**: Finds phrases that might downplay concerns
- **Role-Based Analysis**: Considers the sender's role when evaluating messages
- **Interactive Visualizations**: Shows sentiment flow, risk patterns, and communication gaps
- **Escalation Reports**: Generates detailed reports of flagged messages
- **Downloadable Results**: Allows export of findings for further analysis
