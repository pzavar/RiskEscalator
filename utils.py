import pandas as pd
import re
from datetime import datetime

def format_escalation_report(flagged_messages):
    """
    Format the flagged messages into a readable escalation report.

    Args:
        flagged_messages: List of dictionaries containing flagged messages

    Returns:
        Formatted markdown string for the escalation report
    """
    report = """
<div class="info-card">
<h2>🚨 Risk Escalation Report</h2>
<p style="font-size: 0.9rem; color: #666;">The following messages have been flagged as potentially indicating engineering risks:</p>
"""

    # Group messages by reason
    reason_groups = {}
    for msg in flagged_messages:
        reason = msg.get('reason', 'Unknown reason')
        if reason not in reason_groups:
            reason_groups[reason] = []
        reason_groups[reason].append(msg)

    # For each reason, create a collapsible section with the flagged messages
    for reason, messages in reason_groups.items():
        # Create a sanitized ID for the reason (remove spaces and special chars)
        reason_id = ''.join(e for e in reason if e.isalnum()).lower()

        # Add collapsible section header
        report += f"""
<details>
<summary style="padding: 0.5rem; background-color: #f8f9fa; border-radius: 4px; cursor: pointer; font-weight: bold;">
{reason} ({len(messages)} messages)
</summary>
<div style="margin-top: 0.5rem; padding: 0.5rem; border-left: 3px solid #ddd;">
"""

        # Add each message in a compact card format
        for i, msg in enumerate(messages, 1):
            timestamp = msg.get('timestamp', '')
            if isinstance(timestamp, str):
                try:
                    timestamp = pd.to_datetime(timestamp)
                    formatted_time = timestamp.strftime('%m/%d %H:%M:%S')
                except:
                    formatted_time = timestamp
            else:
                formatted_time = timestamp.strftime('%m/%d %H:%M:%S')

            sender = msg.get('sender', 'Unknown')
            channel = msg.get('channel', 'Unknown')
            message_text = msg.get('message', '')

            # Truncate long messages
            if len(message_text) > 200:
                message_text = message_text[:197] + "..."

            # Create card for each message
            report += f"""
<div style="margin-bottom: 0.7rem; padding: 0.5rem; background-color: #fff; border: 1px solid #eee; border-radius: 4px;">
  <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
    <span style="font-weight: bold; color: #444;">{sender}</span>
    <span style="font-size: 0.8rem; color: #888;">{formatted_time}</span>
  </div>
  <div style="font-size: 0.9rem;">{message_text}</div>
  <div style="font-size: 0.8rem; color: #888; margin-top: 0.3rem;">Channel: {channel}</div>
</div>
"""

        # Close the collapsible section
        report += """
</div>
</details>
"""

    report += "</div>"

    return report

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
        "Thermal Issues": ["thermal", "temperature", "heat", "panel", "warm-up"],
        "Sensor Problems": ["sensor", "reading", "data", "log", "diagnostic"],
        "Anomalies": ["anomaly", "spike", "deviation", "drift", "fluctuation", "weird"],
        "Dismissal": ["not urgent", "minor", "probably nothing", "deemed non-blocking", "not a showstopper"],
        "Persistence": ["still", "again", "not convinced", "hope", "fingers crossed", "ignoring"]
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