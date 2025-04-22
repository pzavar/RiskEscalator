import streamlit as st
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def extract_insights(conversation_data, flagged_messages, analyzed_conversation):
    """
    Extract key insights from the conversation analysis to provide 
    a summarized view of the detected risks.
    
    Args:
        conversation_data: DataFrame containing all messages
        flagged_messages: List of flagged messages
        analyzed_conversation: Dictionary with analysis results
        
    Returns:
        Formatted string with insights
    """
    # Convert flagged_messages to DataFrame if needed
    if flagged_messages and not isinstance(flagged_messages, pd.DataFrame):
        flagged_df = pd.DataFrame(flagged_messages)
    else:
        flagged_df = pd.DataFrame(flagged_messages)
    
    insights = []
    
    # Overview statistics
    insights.append("## Overview")
    
    total_messages = len(conversation_data)
    total_flagged = len(flagged_df) if not flagged_df.empty else 0
    flag_percentage = (total_flagged / total_messages * 100) if total_messages > 0 else 0
    
    insights.append(f"- **Total messages analyzed:** {total_messages}")
    insights.append(f"- **Messages flagged as potential risks:** {total_flagged} ({flag_percentage:.1f}%)")
    
    # Identify key risk themes
    if not flagged_df.empty and 'message' in flagged_df.columns:
        insights.append("\n## Key Risk Themes")
        risk_themes = identify_risk_themes(flagged_df['message'].tolist())
        
        for theme, count in risk_themes:
            insights.append(f"- **{theme}**: Mentioned in {count} flagged messages")
    
    # Communication pattern insights
    insights.append("\n## Communication Pattern Insights")
    
    # Analyze who raised concerns vs who dismissed them
    if not flagged_df.empty and 'sender' in flagged_df.columns and 'reason' in flagged_df.columns:
        concern_raisers = flagged_df[
            flagged_df['reason'].str.contains('concern|raised', case=False)
        ]['sender'].value_counts()
        
        dismissers = flagged_df[
            flagged_df['reason'].str.contains('dismiss|downplay', case=False)
        ]['sender'].value_counts()
        
        if not concern_raisers.empty:
            insights.append("### Who Raised Concerns:")
            for sender, count in concern_raisers.items():
                insights.append(f"- **{sender}**: {count} messages")
        
        if not dismissers.empty:
            insights.append("\n### Who Dismissed Concerns:")
            for sender, count in dismissers.items():
                insights.append(f"- **{sender}**: {count} messages")
    
    # Timeline of risk escalation
    if not flagged_df.empty and 'timestamp' in flagged_df.columns:
        insights.append("\n## Timeline of Risk Evolution")
        
        # Sort flagged messages by timestamp
        sorted_flags = flagged_df.sort_values('timestamp')
        
        # Group by approximate time (minutes)
        sorted_flags['time_group'] = sorted_flags['timestamp'].dt.floor('5min')
        
        # Get a representative message for each time group
        for time_group, group in sorted_flags.groupby('time_group'):
            time_str = time_group.strftime('%H:%M')
            insights.append(f"- **{time_str}**: {len(group)} flagged messages")
            
            # Show the first message from this time group
            if len(group) > 0:
                first_msg = group.iloc[0]
                insights.append(f"  - {first_msg['sender']}: \"{first_msg['message']}\"")
                insights.append(f"  - *Reason flagged*: {first_msg['reason']}")
    
    # Risk severity assessment
    severity = assess_risk_severity(flagged_df if not flagged_df.empty else None, conversation_data)
    insights.append("\n## Risk Severity Assessment")
    insights.append(f"- **Overall severity:** {severity['overall']}")
    insights.append(f"- **Dismissal factor:** {severity['dismissal_factor']}/10")
    insights.append(f"- **Persistence factor:** {severity['persistence_factor']}/10")
    insights.append(f"- **Impact potential:** {severity['impact_potential']}/10")
    
    # Recommendations
    insights.append("\n## Recommendations")
    recommendations = generate_recommendations(flagged_df if not flagged_df.empty else None, severity)
    for rec in recommendations:
        insights.append(f"- {rec}")
    
    return "\n".join(insights)

def identify_risk_themes(messages):
    """
    Identify common themes in the flagged messages.
    
    Args:
        messages: List of message texts
        
    Returns:
        List of tuples with (theme, count)
    """
    if not messages:
        return []
    
    # Combine all messages into a single text
    all_text = ' '.join(messages)
    
    # Define theme keywords
    themes = {
        "Thermal Issues": ["thermal", "temperature", "heat", "hot", "warm", "panel"],
        "Sensor Problems": ["sensor", "reading", "data", "log", "measurement", "diagnostic"],
        "Anomalies": ["anomaly", "spike", "deviation", "drift", "fluctuation", "weird"],
        "Communication Issues": ["not prioritize", "ignoring", "dismissed", "overlooked", "defer"],
        "System Concerns": ["system", "electrical", "hardware", "software", "component"]
    }
    
    # Count occurrences of each theme
    theme_counts = []
    
    for theme, keywords in themes.items():
        count = sum(1 for keyword in keywords if keyword.lower() in all_text.lower())
        if count > 0:
            theme_counts.append((theme, count))
    
    # Sort by count
    theme_counts.sort(key=lambda x: x[1], reverse=True)
    
    return theme_counts

def assess_risk_severity(flagged_df, conversation_data):
    """
    Assess the severity of the detected risks.
    
    Args:
        flagged_df: DataFrame with flagged messages
        conversation_data: Full conversation data
        
    Returns:
        Dictionary with severity assessment
    """
    # Default values if no data
    if flagged_df is None or flagged_df.empty:
        return {
            'overall': "Low",
            'dismissal_factor': 2,
            'persistence_factor': 1,
            'impact_potential': 3
        }
    
    # Calculate dismissal factor (how often concerns were dismissed)
    dismissal_count = sum(1 for reason in flagged_df['reason'] if 'dismiss' in reason.lower() or 'downplay' in reason.lower())
    dismissal_factor = min(10, int((dismissal_count / len(flagged_df)) * 10) + 3)
    
    # Calculate persistence factor (were concerns raised multiple times)
    # Count unique senders who raised concerns
    concern_raisers = flagged_df[
        flagged_df['reason'].str.contains('concern|raised', case=False)
    ]['sender'].nunique()
    
    # Count messages with words indicating persistence
    persistence_words = ['still', 'again', 'continues', 'persist', 'repeat']
    persistence_count = sum(
        1 for msg in flagged_df['message'] 
        if any(word in msg.lower() for word in persistence_words)
    )
    
    persistence_factor = min(10, concern_raisers * 2 + persistence_count)
    
    # Calculate impact potential
    # Look for words indicating severity
    severity_words = {
        'high': ['critical', 'serious', 'significant', 'major', 'important', 'dangerous'],
        'medium': ['concerning', 'notable', 'unusual', 'unexpected', 'strange'],
        'low': ['minor', 'small', 'slight', 'tiny', 'little']
    }
    
    # Count severity indications
    high_count = sum(
        1 for msg in flagged_df['message'] 
        if any(word in msg.lower() for word in severity_words['high'])
    )
    
    medium_count = sum(
        1 for msg in flagged_df['message'] 
        if any(word in msg.lower() for word in severity_words['medium'])
    )
    
    low_count = sum(
        1 for msg in flagged_df['message'] 
        if any(word in msg.lower() for word in severity_words['low'])
    )
    
    # Calculate impact score
    impact_potential = min(10, (high_count * 3 + medium_count * 2 + low_count) + 2)
    
    # Calculate overall severity
    overall_score = (dismissal_factor + persistence_factor + impact_potential) / 3
    
    if overall_score >= 7:
        overall = "High"
    elif overall_score >= 4:
        overall = "Medium"
    else:
        overall = "Low"
    
    return {
        'overall': overall,
        'dismissal_factor': dismissal_factor,
        'persistence_factor': persistence_factor,
        'impact_potential': impact_potential
    }

def generate_recommendations(flagged_df, severity):
    """
    Generate recommendations based on the analysis.
    
    Args:
        flagged_df: DataFrame with flagged messages
        severity: Dictionary with severity assessment
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Basic recommendations that always apply
    recommendations.append("Review all flagged messages and investigate potential issues")
    recommendations.append("Ensure engineering concerns are properly acknowledged and evaluated")
    
    # Add severity-specific recommendations
    if severity['overall'] == "High":
        recommendations.append("**URGENT**: Immediate follow-up required on all flagged issues")
        recommendations.append("Consider implementing a more structured risk reporting process")
        recommendations.append("Review decision-making hierarchy to ensure technical concerns aren't overlooked")
        
    elif severity['overall'] == "Medium":
        recommendations.append("Schedule a follow-up discussion on the flagged issues within 24-48 hours")
        recommendations.append("Document all potential concerns and track resolution")
        
    else:  # Low
        recommendations.append("Monitor the situation for any changes or escalations")
    
    # Add recommendations based on specific factors
    if severity['dismissal_factor'] >= 6:
        recommendations.append("Address potential communication issues between leadership and engineering teams")
    
    if severity['persistence_factor'] >= 6:
        recommendations.append("Take note of recurring concerns that may indicate a persistent problem")
    
    if severity['impact_potential'] >= 6:
        recommendations.append("Evaluate potential impact if the identified issues were to manifest")
    
    return recommendations
