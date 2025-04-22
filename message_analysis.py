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
    a summarized view of the detected risks focused on executive needs.
    
    Args:
        conversation_data: DataFrame containing all messages
        flagged_messages: List of flagged messages
        analyzed_conversation: Dictionary with analysis results
        
    Returns:
        Formatted HTML string with executive-focused insights
    """
    # Convert flagged_messages to DataFrame if needed
    if flagged_messages and not isinstance(flagged_messages, pd.DataFrame):
        flagged_df = pd.DataFrame(flagged_messages)
    else:
        flagged_df = pd.DataFrame(flagged_messages)
    
    insights = []
    
    # Calculate key metrics
    total_messages = len(conversation_data)
    total_flagged = len(flagged_df) if not flagged_df.empty else 0
    flag_percentage = (total_flagged / total_messages * 100) if total_messages > 0 else 0
    
    # Risk severity assessment
    severity = assess_risk_severity(flagged_df if not flagged_df.empty else None, conversation_data)
    severity_color = "#d13438" if severity['overall'] == "High" else "#ff8c00" if severity['overall'] == "Medium" else "#107c10"
    
    # Executive Summary Card
    insights.append(f"""
    <div style="background-color: {severity_color}10; padding: 1.5rem; border-left: 5px solid {severity_color}; border-radius: 5px; margin-bottom: 2rem;">
        <h3 style="margin-top: 0; color: {severity_color};">{severity['overall']} Risk Severity</h3>
        <p>The analysis identified {total_flagged} messages ({flag_percentage:.1f}%) containing potential risk indicators 
        that may require leadership attention.</p>
        
        <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem;">
            <div style="flex: 1; min-width: 200px; background-color: white; padding: 1rem; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: {severity_color};">Dismissal Factor</h4>
                <div style="font-size: 1.5rem; font-weight: bold;">{severity['dismissal_factor']}/10</div>
                <p style="margin-bottom: 0; font-size: 0.9rem; color: #666;">Measures how often concerns were downplayed</p>
            </div>
            
            <div style="flex: 1; min-width: 200px; background-color: white; padding: 1rem; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: {severity_color};">Persistence Factor</h4>
                <div style="font-size: 1.5rem; font-weight: bold;">{severity['persistence_factor']}/10</div>
                <p style="margin-bottom: 0; font-size: 0.9rem; color: #666;">How persistently concerns were raised</p>
            </div>
            
            <div style="flex: 1; min-width: 200px; background-color: white; padding: 1rem; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: {severity_color};">Impact Potential</h4>
                <div style="font-size: 1.5rem; font-weight: bold;">{severity['impact_potential']}/10</div>
                <p style="margin-bottom: 0; font-size: 0.9rem; color: #666;">Potential business impact if issues materialize</p>
            </div>
        </div>
    </div>
    """)
    
    # Business Impact Assessment - what executives care about most
    insights.append("""
    <div style="margin-bottom: 2rem;">
        <h3 style="color: #0078d4; margin-bottom: 1rem;">Business Impact Assessment</h3>
    """)
    
    # Identify key risk themes for business impact
    if not flagged_df.empty and 'message' in flagged_df.columns:
        risk_themes = identify_risk_themes(flagged_df['message'].tolist())
        
        if risk_themes:
            primary_theme = risk_themes[0][0] if risk_themes else "Unknown"
            primary_count = risk_themes[0][1] if risk_themes else 0
            
            # Business impact narrative based on the most common theme
            impact_narratives = {
                "Thermal Issues": "Potential thermal issues may impact hardware reliability and could lead to system failures if not addressed.",
                "Sensor Problems": "Sensor malfunctions could result in unreliable data collection, potentially affecting decision quality.",
                "Anomalies": "Detected anomalies suggest irregular system behavior that may compromise operational stability.",
                "System Concerns": "System concerns indicate potential infrastructure weaknesses that could affect service delivery."
            }
            
            impact_narrative = impact_narratives.get(primary_theme, "The identified issues require further technical investigation to assess potential business impact.")
            
            insights.append(f"""
            <div style="display: flex; margin-bottom: 1.5rem;">
                <div style="flex: 2; margin-right: 1rem;">
                    <h4 style="margin-top: 0;">Primary Risk Area: {primary_theme}</h4>
                    <p>{impact_narrative}</p>
                </div>
                <div style="flex: 1; background-color: #f8f9fa; padding: 1rem; border-radius: 5px;">
                    <h4 style="margin-top: 0;">Risk Distribution</h4>
                    <ul style="margin-bottom: 0;">
            """)
            
            for theme, count in risk_themes:
                insights.append(f"<li><strong>{theme}</strong>: {count} mentions</li>")
                
            insights.append("""
                    </ul>
                </div>
            </div>
            """)
    
    # Key People Analysis - focus on decision makers
    if not flagged_df.empty and 'sender' in flagged_df.columns and 'reason' in flagged_df.columns:
        concern_raisers = flagged_df[
            flagged_df['reason'].str.contains('concern|raised', case=False)
        ]['sender'].value_counts()
        
        dismissers = flagged_df[
            flagged_df['reason'].str.contains('dismiss|downplay', case=False)
        ]['sender'].value_counts()
        
        if not concern_raisers.empty or not dismissers.empty:
            insights.append("""
            <h4 style="margin-bottom: 1rem;">Key Team Members Involved</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem;">
            """)
            
            if not concern_raisers.empty:
                insights.append("""
                <div style="flex: 1; min-width: 300px; background-color: #fef9f9; padding: 1rem; border-left: 3px solid #d13438; border-radius: 5px;">
                    <h5 style="margin-top: 0; color: #d13438;">Raising Technical Concerns</h5>
                    <ul style="margin-bottom: 0;">
                """)
                for sender, count in concern_raisers.items():
                    insights.append(f"<li><strong>{sender}</strong>: {count} flagged messages</li>")
                insights.append("</ul></div>")
            
            if not dismissers.empty:
                insights.append("""
                <div style="flex: 1; min-width: 300px; background-color: #fff9f0; padding: 1rem; border-left: 3px solid #ff8c00; border-radius: 5px;">
                    <h5 style="margin-top: 0; color: #ff8c00;">Potentially Downplaying Risks</h5>
                    <ul style="margin-bottom: 0;">
                """)
                for sender, count in dismissers.items():
                    insights.append(f"<li><strong>{sender}</strong>: {count} flagged messages</li>")
                insights.append("</ul></div>")
                
            insights.append("</div>")
    
    insights.append("</div>")  # Close business impact section
    
    # Executive Recommendations
    insights.append("""
    <div style="background-color: #f0f8ff; padding: 1.5rem; border-radius: 5px; margin-bottom: 2rem;">
        <h3 style="margin-top: 0; color: #0078d4;">Leadership Action Items</h3>
    """)
    
    recommendations = generate_recommendations(flagged_df if not flagged_df.empty else None, severity)
    insights.append("<ol style='margin-bottom: 0;'>")
    for rec in recommendations:
        insights.append(f"<li>{rec}</li>")
    insights.append("</ol></div>")
    
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
    Generate executive-focused recommendations based on the analysis.
    
    Args:
        flagged_df: DataFrame with flagged messages
        severity: Dictionary with severity assessment
        
    Returns:
        List of recommendation strings formatted for executive audience
    """
    recommendations = []
    
    # Severity-specific action items with business impact focus
    if severity['overall'] == "High":
        recommendations.append("<strong>Immediate action required:</strong> Schedule an engineering review within 24 hours to assess potential risks to project/product delivery timelines.")
        recommendations.append("<strong>Establish a technical task force</strong> led by engineers who raised concerns to investigate and report findings directly to leadership.")
        recommendations.append("<strong>Implement immediate safeguards</strong> against the identified technical risks to mitigate potential business impact.")
        
    elif severity['overall'] == "Medium":
        recommendations.append("<strong>Prioritize technical review:</strong> Schedule a focused discussion with engineering team within 72 hours to evaluate identified concerns.")
        recommendations.append("<strong>Create a structured risk tracking system</strong> to ensure technical concerns receive appropriate visibility and follow-up.")
        recommendations.append("<strong>Conduct a communication review</strong> to ensure engineering feedback is properly escalated through appropriate channels.")
        
    else:  # Low
        recommendations.append("<strong>Monitor situation:</strong> Ensure regular check-ins on these identified areas in upcoming project status meetings.")
        recommendations.append("<strong>Document concerns raised</strong> in the project risk register for continued visibility.")
    
    # Add recommendations based on specific factors - tailored for business context
    if severity['dismissal_factor'] >= 6:
        recommendations.append("<strong>Review decision-making processes:</strong> Evaluate how technical input is incorporated into leadership decisions to strengthen engineering-management alignment.")
    
    if severity['persistence_factor'] >= 6:
        recommendations.append("<strong>Technical deep dive required:</strong> When engineers repeatedly raise the same concerns, schedule dedicated technical reviews to fully understand potential impacts.")
    
    if severity['impact_potential'] >= 6:
        recommendations.append("<strong>Impact assessment needed:</strong> Evaluate potential business consequences if these technical issues were to materialize, including customer impact, revenue implications, and reputational risks.")
    
    # Add a general leadership recommendation that always applies
    recommendations.append("<strong>Leadership communication strategy:</strong> Ensure engineers feel heard and valued when raising technical concerns to strengthen team culture and improve early risk detection.")
    
    return recommendations
