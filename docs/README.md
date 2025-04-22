
# Risk Detection System Documentation

Welcome to the Risk Detection System documentation. This system analyzes Slack conversations to identify potential risks that might be buried or downplayed in engineering discussions.

## Available Documentation

- [Sentiment Analysis](sentiment_analysis.md) - How the system analyzes message sentiment to detect risks
- [System Overview](system_overview.md) - General system architecture and workflow
- [Risk Detection Algorithms](risk_detection.md) - Details of the risk detection algorithms and keywords

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
