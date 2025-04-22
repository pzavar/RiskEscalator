import pandas as pd
import re
from datetime import datetime

def format_escalation_report(flagged_messages):
    """
    Format the flagged messages into a Markdown escalation report with executive styling.
    
    Args:
        flagged_messages: List of dictionaries with flagged message information
        
    Returns:
        Formatted Markdown string with executive-friendly styling
    """
    if not flagged_messages:
        return "No messages were flagged for escalation."
    
    report_lines = []
    report_lines.append(f"""
    <div style="margin-bottom: 2rem;">
        <h2 style="color: #0078d4; margin-bottom: 0.5rem;">Risk Escalation Report</h2>
        <p style="color: #666; margin-bottom: 1rem;">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Total flagged messages: {len(flagged_messages)}</p>
    </div>
    """)
    
    # Add a summary of key findings first - executives want the summary at the top
    report_lines.append("""
    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 5px; margin-bottom: 2rem;">
        <h3 style="margin-top: 0; color: #0078d4;">Executive Summary of Key Findings</h3>
    """)
    
    # Identify common themes in the flagged messages
    themes = identify_themes_in_messages(flagged_messages)
    
    if themes:
        report_lines.append("<ul style='margin-bottom: 1.5rem;'>")
        for theme, count in themes:
            report_lines.append(f"<li><strong>{theme}</strong>: Mentioned in {count} flagged messages</li>")
        report_lines.append("</ul>")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(flagged_messages)
    
    # Add conversation flow insights - who raised vs dismissed concerns
    concern_raisers = df[df['reason'].str.contains('concern|raised', case=False)]['sender'].value_counts()
    dismissers = df[df['reason'].str.contains('dismiss|downplay', case=False)]['sender'].value_counts()
    
    if not concern_raisers.empty or not dismissers.empty:
        report_lines.append("""
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
        """)
        
        if not concern_raisers.empty:
            report_lines.append("""
            <div style="flex: 1; min-width: 250px; margin-right: 1rem;">
                <h4 style="color: #d13438; margin-bottom: 0.5rem;">Team Members Raising Concerns:</h4>
                <ul>
            """)
            for sender, count in concern_raisers.items():
                report_lines.append(f"<li><strong>{sender}</strong>: {count} messages</li>")
            report_lines.append("</ul></div>")
        
        if not dismissers.empty:
            report_lines.append("""
            <div style="flex: 1; min-width: 250px;">
                <h4 style="color: #ff8c00; margin-bottom: 0.5rem;">Team Members Potentially Downplaying Risks:</h4>
                <ul>
            """)
            for sender, count in dismissers.items():
                report_lines.append(f"<li><strong>{sender}</strong>: {count} messages</li>")
            report_lines.append("</ul></div>")
            
        report_lines.append("</div>")  # Close flex container
    
    report_lines.append("</div>")  # Close summary container
    
    # Timeline of flagged messages
    report_lines.append("""
    <h3 style="color: #0078d4; margin: 2rem 0 1rem 0;">Timeline of Risk Indicators</h3>
    <div style="border-left: 3px solid #0078d4; padding-left: 1.5rem;">
    """)
    
    # Sort messages chronologically
    sorted_messages = sorted(flagged_messages, key=lambda x: x['timestamp'])
    
    # Group messages by approximate time (15-minute windows) for a cleaner report
    from datetime import datetime, timedelta
    
    time_groups = {}
    for msg in sorted_messages:
        timestamp = msg['timestamp']
        if isinstance(timestamp, pd.Timestamp):
            time_key = timestamp.floor('15min')
        else:
            try:
                dt = pd.to_datetime(timestamp)
                time_key = dt.floor('15min')
            except:
                time_key = timestamp  # Just use as is if we can't parse
                
        if time_key not in time_groups:
            time_groups[time_key] = []
        time_groups[time_key].append(msg)
    
    # Format each time group
    for time_key, messages in time_groups.items():
        if isinstance(time_key, pd.Timestamp):
            time_str = time_key.strftime('%H:%M')
        else:
            time_str = str(time_key)
            
        report_lines.append(f"""
        <div style="margin-bottom: 2rem;">
            <h4 style="color: #555; margin-bottom: 0.5rem;">{time_str} - {len(messages)} message(s)</h4>
        """)
        
        for msg in messages:
            # Determine severity color based on reason
            severity_color = "#d13438" if "dismissed" in msg['reason'].lower() or "downplay" in msg['reason'].lower() else "#ff8c00"
            
            report_lines.append(f"""
            <div style="background-color: {severity_color}11; padding: 1rem; border-left: 3px solid {severity_color}; 
                 border-radius: 3px; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <div><strong>{msg['sender']}</strong> in {msg['channel']}</div>
                </div>
                <div style="background-color: white; padding: 0.75rem; border-radius: 3px; margin-bottom: 0.5rem;">
                    {msg['message']}
                </div>
                <div style="font-size: 0.9rem; color: {severity_color};">
                    <strong>Risk indicator:</strong> {msg['reason']}
                </div>
            </div>
            """)
            
        report_lines.append("</div>")
    
    report_lines.append("</div>")  # Close timeline container
    
    # Add recommendations section
    report_lines.append("""
    <h3 style="color: #0078d4; margin: 2rem 0 1rem 0;">Recommended Actions</h3>
    <div style="background-color: #f0f8ff; padding: 1.5rem; border-radius: 5px; border-left: 3px solid #0078d4;">
        <ol>
            <li>Review all flagged messages in their context with the engineering team</li>
            <li>Schedule a focused discussion with team members who raised persistent concerns</li>
            <li>Implement a more structured approach to tracking technical concerns raised by engineers</li>
            <li>Consider creating a technical risk register for this project</li>
        </ol>
    </div>
    """)
    
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
