
# Risk Detection System Documentation

<video width="100%" controls>
  <source src="risk detector demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Welcome to the Risk Detection System documentation. This system analyzes Slack conversations to identify potential risks that might be buried or downplayed in engineering discussions.

## Available Documentation

- [Sentiment Analysis](sentiment_analysis.md) - How the system analyzes message sentiment to detect risks
- [System Overview](system_overview.md) - General system architecture and workflow
- [Risk Detection Algorithms](risk_detection.md) - Details of the risk detection algorithms and keywords
- [Data Processing Flow](data_processing_flow.md) - Comprehensive documentation of the CSV upload and processing workflow

## Getting Started

To use the Risk Detection System:

1. Upload a CSV file containing Slack conversation data
2. Click "Analyze Conversation" to process the data
3. View the results in the Escalation Report, Visualizations, and Insights tabs

## Required Data Format

The CSV file must contain the following columns:
- `timestamp`: When the message was sent
- `sender`: Who sent the message
- `channel`: Which channel the message was sent in
- `message`: The content of the message

## User Interface

The system presents results in three organized tabs:

1. **Escalation Report**: A detailed report of all flagged messages with context
2. **Visualizations**: Interactive charts showing risk patterns and communication flow
3. **Insights**: Key findings, risk severity assessment, and actionable recommendations

For a complete understanding of how your data is processed when uploaded, see the [Data Processing Flow](data_processing_flow.md) documentation.
