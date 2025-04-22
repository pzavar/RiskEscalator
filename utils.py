import pandas as pd
import re
from datetime import datetime

def format_escalation_report(flagged_messages):
    """
    Format the flagged messages into a Markdown escalation report.
    
    Args:
        flagged_messages: List of dictionaries with flagged message information
        
    Returns:
        Formatted Markdown string
    """
    if not flagged_messages:
        return "No messages were flagged for escalation."
    
    report_lines = []
    report_lines.append("# Risk Escalation Report")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total messages flagged: {len(flagged_messages)}")
    report_lines.append("\n---\n")
    
    # Group flagged messages by sender
    df = pd.DataFrame(flagged_messages)
    sender_groups = df.groupby('sender')
    
    for sender, group in sender_groups:
        report_lines.append(f"## Messages from {sender}")
        
        for _, row in group.iterrows():
            timestamp = row['timestamp']
            if isinstance(timestamp, pd.Timestamp):
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = str(timestamp)
                
            report_lines.append(f"### {time_str} in {row['channel']}")
            report_lines.append(f"> {row['message']}")
            report_lines.append(f"**Reason flagged:** {row['reason']}")
            report_lines.append("\n---\n")
    
    # Add a summary of key findings
    report_lines.append("## Summary of Key Findings")
    
    # Identify common themes in the flagged messages
    themes = identify_themes_in_messages(flagged_messages)
    
    for theme, count in themes:
        report_lines.append(f"- **{theme}**: Mentioned in {count} flagged messages")
    
    # Add conversation flow insights
    report_lines.append("\n## Conversation Flow Insights")
    
    # Identify who raised concerns vs who dismissed them
    concern_raisers = df[df['reason'].str.contains('concern|raised', case=False)]['sender'].value_counts()
    dismissers = df[df['reason'].str.contains('dismiss|downplay', case=False)]['sender'].value_counts()
    
    if not concern_raisers.empty:
        report_lines.append("### Who Raised Concerns:")
        for sender, count in concern_raisers.items():
            report_lines.append(f"- **{sender}**: {count} messages")
    
    if not dismissers.empty:
        report_lines.append("\n### Who Dismissed Concerns:")
        for sender, count in dismissers.items():
            report_lines.append(f"- **{sender}**: {count} messages")
    
    return "\n".join(report_lines)

def identify_themes_in_messages(flagged_messages):
    """
    Identify common themes in the flagged messages.
    
    Args:
        flagged_messages: List of dictionaries with flagged message information
        
    Returns:
        List of tuples with (theme, count)
    """
    # Define theme keywords
    themes = {
        "Thermal Issues": ["thermal", "temperature", "heat", "panel"],
        "Sensor Problems": ["sensor", "reading", "data", "log"],
        "Anomalies": ["anomaly", "spike", "deviation", "drift"],
        "Dismissal": ["not urgent", "minor", "probably nothing", "deemed non-blocking"],
        "Persistence": ["still", "again", "not convinced", "hope"]
    }
    
    # Count occurrences of each theme
    theme_counts = {}
    
    for message_info in flagged_messages:
        message = message_info['message'].lower()
        
        for theme, keywords in themes.items():
            if any(keyword.lower() in message for keyword in keywords):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    # Convert to list of tuples and sort by count
    theme_list = [(theme, count) for theme, count in theme_counts.items()]
    theme_list.sort(key=lambda x: x[1], reverse=True)
    
    return theme_list

def clean_text(text):
    """
    Clean text for better processing.
    
    Args:
        text: Text string to clean
        
    Returns:
        Cleaned text string
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def format_timestamp(timestamp):
    """
    Format timestamp consistently.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(timestamp, str):
        try:
            dt = pd.to_datetime(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return timestamp
    else:
        return str(timestamp)
