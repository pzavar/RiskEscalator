import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import re

# Download necessary NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Risk-related keywords
RISK_KEYWORDS = [
    'spike', 'anomaly', 'weird', 'thermal deviation', 'not urgent but', 
    'deviation', 'unusual', 'abnormal', 'drift', 'fluctuation', 'issue',
    'bug', 'glitch', 'error', 'warning', 'concern', 'problem', 'malfunction',
    'failure', 'fault', 'defect', 'inconsistent', 'unexpected', 'irregular'
]

# Dismissive language patterns
DISMISSIVE_PATTERNS = [
    'not a big deal', 'probably nothing', 'don\'t worry', 'not critical',
    'no need to', 'minor', 'not urgent', 'can ignore', 'non-blocking',
    'not a showstopper', 'not alarming', 'within tolerance', 'noise',
    'harmless', 'nothing to worry about', 'not a concern', 'deemed non-blocking',
    'no criticals', 'not prioritize', 'all clear', 'no red flags'
]

# Leadership and decision-maker roles
LEADERSHIP_ROLES = ['PM_Lead', 'Director', 'QA_Tech', 'Systems_Admin']

def detect_risks(df):
    """
    Main function to detect risks in the conversation.
    
    Args:
        df: DataFrame containing the Slack messages
        
    Returns:
        List of dictionaries with flagged messages and their reasons
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis to each message
    df_copy['sentiment'] = df_copy['message'].apply(
        lambda x: sia.polarity_scores(x)
    )
    df_copy['compound_sentiment'] = df_copy['sentiment'].apply(lambda x: x['compound'])
    
    # Check for risk keywords in each message
    df_copy['contains_risk_word'] = df_copy['message'].apply(
        lambda x: any(keyword.lower() in x.lower() for keyword in RISK_KEYWORDS)
    )
    
    # Check for dismissive language in each message
    df_copy['is_dismissive'] = df_copy['message'].apply(
        lambda x: any(re.search(r'\b' + re.escape(pattern.lower()) + r'\b', x.lower()) for pattern in DISMISSIVE_PATTERNS)
    )
    
    # Check if the sender is a leadership role
    df_copy['is_leadership'] = df_copy['sender'].isin(LEADERSHIP_ROLES)
    
    # Identify messages that might be downplaying risks
    df_copy['downplaying_risk'] = (
        (df_copy['contains_risk_word'] & df_copy['is_dismissive']) | 
        (df_copy['contains_risk_word'] & (df_copy['compound_sentiment'] > 0) & df_copy['is_leadership'])
    )
    
    # Find message clusters related to risks
    risk_clusters = find_risk_clusters(df_copy)
    
    # Identify messages that raise concerns but have been dismissed
    flagged_messages = identify_dismissed_concerns(df_copy, risk_clusters)
    
    # Sort the flagged messages by timestamp
    flagged_messages.sort(key=lambda x: x['timestamp'])
    
    return flagged_messages

def find_risk_clusters(df):
    """
    Find clusters of messages that are discussing the same risk issue.
    
    Args:
        df: DataFrame with processed message data
        
    Returns:
        List of clusters (each cluster is a list of message indices)
    """
    # Get messages that contain risk words
    risk_messages = df[df['contains_risk_word']]
    
    if len(risk_messages) <= 1:
        return [[i] for i in risk_messages.index] if len(risk_messages) > 0 else []
    
    # Create a corpus of just the risk-related messages
    corpus = risk_messages['message'].tolist()
    
    # Use CountVectorizer to convert the text to a matrix of token counts
    vectorizer = CountVectorizer(stop_words='english', min_df=1, binary=True)
    X = vectorizer.fit_transform(corpus)
    
    # Normalize the vectors
    X_normalized = normalize(X)
    
    # Calculate cosine similarity
    similarity_matrix = (X_normalized @ X_normalized.T).toarray()
    
    # Set threshold for similarity
    similarity_threshold = 0.3
    
    # Create clusters
    clusters = []
    visited = set()
    
    for i in range(len(corpus)):
        if i in visited:
            continue
            
        cluster = [risk_messages.index[i]]
        visited.add(i)
        
        for j in range(len(corpus)):
            if i != j and j not in visited and similarity_matrix[i, j] > similarity_threshold:
                cluster.append(risk_messages.index[j])
                visited.add(j)
                
        clusters.append(cluster)
    
    return clusters

def identify_dismissed_concerns(df, risk_clusters):
    """
    Identify messages that raise concerns but have been dismissed.
    
    Args:
        df: DataFrame with processed message data
        risk_clusters: Clusters of related risk messages
        
    Returns:
        List of dictionaries with flagged messages and reasons
    """
    flagged_messages = []
    
    # Process each risk cluster
    for cluster in risk_clusters:
        cluster_messages = df.loc[cluster].sort_values('timestamp')
        
        # Check if concerns were raised by engineers but dismissed by leadership
        concerns_raised = False
        concerns_dismissed = False
        
        for idx, row in cluster_messages.iterrows():
            # Check if this is a concern being raised (by non-leadership)
            if row['contains_risk_word'] and not row['is_leadership']:
                # Made the sentiment check less strict to catch more subtle concerns
                concerns_raised = True
                
                # Add this message to the flagged list
                flagged_messages.append({
                    'timestamp': row['timestamp'],
                    'sender': row['sender'],
                    'channel': row['channel'],
                    'message': row['message'],
                    'reason': "Raised concern about a potential risk issue"
                })
            
            # Check if a leader is dismissing concerns
            elif row['is_leadership'] and (row['is_dismissive'] or (row['compound_sentiment'] > 0 and any(keyword.lower() in row['message'].lower() for keyword in RISK_KEYWORDS))):
                concerns_dismissed = True
                
                # Add this message to the flagged list
                flagged_messages.append({
                    'timestamp': row['timestamp'],
                    'sender': row['sender'],
                    'channel': row['channel'],
                    'message': row['message'],
                    'reason': "Leadership potentially downplaying or dismissing concerns"
                })
        
        # If concerns were raised but not adequately addressed, flag other related messages
        if concerns_raised and len(cluster_messages) > 1:
            # Look for messages where engineers express further concerns or uncertainty
            for idx, row in cluster_messages.iterrows():
                if not row['is_leadership'] and not any(msg['timestamp'] == row['timestamp'] for msg in flagged_messages):
                    # More lenient detection of expressions of doubt, frustration, or persistence
                    if ('still' in row['message'].lower() or 
                        'not convinced' in row['message'].lower() or 
                        'hope' in row['message'].lower() or 
                        'guess' in row['message'].lower() or
                        'documenting' in row['message'].lower() or
                        'fingers crossed' in row['message'].lower() or
                        '...' in row['message']):
                        
                        flagged_messages.append({
                            'timestamp': row['timestamp'],
                            'sender': row['sender'],
                            'channel': row['channel'],
                            'message': row['message'],
                            'reason': "Continued concern or doubt expressed after initial discussion"
                        })
    
    return flagged_messages

def analyze_conversation(df):
    """
    Analyze the conversation for overall patterns.
    
    Args:
        df: DataFrame containing the Slack messages
        
    Returns:
        Dictionary with analysis results
    """
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment for all messages
    sentiments = df['message'].apply(lambda x: sia.polarity_scores(x)['compound']).tolist()
    
    # Calculate various statistics and patterns
    results = {
        'total_messages': len(df),
        'messages_per_sender': df['sender'].value_counts().to_dict(),
        'messages_per_channel': df['channel'].value_counts().to_dict(),
        'avg_sentiment': np.mean(sentiments),
        'sentiment_trend': sentiments,
        'risk_keywords_found': sum(1 for msg in df['message'] if any(keyword.lower() in msg.lower() for keyword in RISK_KEYWORDS)),
        'dismissive_language_count': sum(1 for msg in df['message'] if any(pattern.lower() in msg.lower() for pattern in DISMISSIVE_PATTERNS)),
        'leadership_message_count': sum(df['sender'].isin(LEADERSHIP_ROLES)),
        'timeline': df['timestamp'].tolist()
    }
    
    # Analyze conversation flow to detect disconnects
    results['communication_gaps'] = detect_communication_gaps(df)
    
    return results

def detect_communication_gaps(df):
    """
    Detect potential communication gaps in the conversation.
    
    Args:
        df: DataFrame containing the Slack messages
        
    Returns:
        List of potential communication gap instances
    """
    gaps = []
    
    # Find instances where concerns were raised but not addressed
    # Group messages by approximate time windows (5 minute intervals)
    df['time_window'] = df['timestamp'].dt.floor('5min')
    
    grouped = df.groupby('time_window')
    
    for time_window, group in grouped:
        # Check if engineers raised concerns in this window
        engineer_concerns = group[
            (~group['sender'].isin(LEADERSHIP_ROLES)) & 
            (group['message'].str.contains('|'.join(RISK_KEYWORDS), case=False))
        ]
        
        # Check if there were leadership responses that addressed these concerns
        if len(engineer_concerns) > 0:
            leadership_responses = group[group['sender'].isin(LEADERSHIP_ROLES)]
            
            # If there were no leadership responses or they were dismissive
            if len(leadership_responses) == 0 or any(leadership_responses['message'].str.contains('|'.join(DISMISSIVE_PATTERNS), case=False)):
                gaps.append({
                    'time_window': time_window,
                    'concerns': engineer_concerns.to_dict('records'),
                    'responses': leadership_responses.to_dict('records') if len(leadership_responses) > 0 else []
                })
    
    return gaps
