import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

def generate_visualizations(conversation_data, flagged_messages, analyzed_conversation):
    """
    Generate visualizations for the risk detection results.
    
    Args:
        conversation_data: DataFrame containing all conversation messages
        flagged_messages: List of flagged messages with reasons
        analyzed_conversation: Dictionary containing analysis results
    """
    # Convert flagged_messages to DataFrame if it's not empty
    if flagged_messages:
        flagged_df = pd.DataFrame(flagged_messages)
        flagged_df['timestamp'] = pd.to_datetime(flagged_df['timestamp'])
    else:
        flagged_df = pd.DataFrame(columns=['timestamp', 'sender', 'channel', 'message', 'reason'])
    
    # Create two columns for the visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Timeline visualization of all messages with flagged messages highlighted
        create_timeline_visualization(conversation_data, flagged_df)
    
    with col2:
        # Sender distribution showing who is raising concerns vs dismissing them
        create_sender_distribution(conversation_data, flagged_df)
    
    # Message sentiment flow
    st.subheader("Message Sentiment Flow")
    create_sentiment_flow(conversation_data)
    
    # Channel distribution of flagged messages
    if not flagged_df.empty:
        st.subheader("Distribution of Flagged Messages by Channel")
        create_channel_distribution(flagged_df)
    
    # Risk word occurrence heatmap
    st.subheader("Risk Keyword Occurrence")
    create_risk_keyword_heatmap(conversation_data)

def create_timeline_visualization(all_messages, flagged_messages):
    """
    Create a timeline visualization showing all messages with flagged ones highlighted.
    
    Args:
        all_messages: DataFrame containing all conversation messages
        flagged_messages: DataFrame containing flagged messages
    """
    st.subheader("Conversation Timeline with Flagged Messages")
    
    # Create a copy of the all_messages DataFrame and add a flagged column
    timeline_df = all_messages.copy()
    timeline_df['flagged'] = timeline_df['timestamp'].isin(flagged_messages['timestamp'])
    timeline_df['flag_label'] = 'Normal Message'
    
    # Mark flagged messages
    if not flagged_messages.empty:
        for idx, row in timeline_df[timeline_df['flagged']].iterrows():
            # Find the reason for this flagged message
            flag_reason = flagged_messages[flagged_messages['timestamp'] == row['timestamp']]['reason'].values
            if len(flag_reason) > 0:
                timeline_df.at[idx, 'flag_label'] = f'Flagged: {flag_reason[0]}'
    
    # Create the timeline visualization
    fig = px.scatter(
        timeline_df,
        x='timestamp',
        y='sender',
        color='flag_label',
        hover_data=['message', 'channel'],
        size_max=10,
        color_discrete_map={
            'Normal Message': 'gray',
            **{col: 'red' for col in timeline_df['flag_label'].unique() if col != 'Normal Message'}
        }
    )
    
    fig.update_layout(
        title="Conversation Timeline",
        xaxis_title="Time",
        yaxis_title="Sender",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_sender_distribution(all_messages, flagged_messages):
    """
    Create a visualization showing the distribution of messages by sender,
    highlighting those who are raising concerns vs dismissing them.
    
    Args:
        all_messages: DataFrame containing all conversation messages
        flagged_messages: DataFrame containing flagged messages
    """
    st.subheader("Message Distribution by Sender")
    
    # Count total messages by sender
    sender_counts = all_messages['sender'].value_counts().reset_index()
    sender_counts.columns = ['sender', 'total_messages']
    
    # Count flagged messages by sender (if any)
    if not flagged_messages.empty:
        flagged_counts = flagged_messages['sender'].value_counts().reset_index()
        flagged_counts.columns = ['sender', 'flagged_messages']
        
        # Merge the counts
        sender_stats = pd.merge(sender_counts, flagged_counts, on='sender', how='left')
        sender_stats['flagged_messages'] = sender_stats['flagged_messages'].fillna(0)
        sender_stats['normal_messages'] = sender_stats['total_messages'] - sender_stats['flagged_messages']
        
        # Create a stacked bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sender_stats['sender'],
            y=sender_stats['normal_messages'],
            name='Normal Messages',
            marker_color='lightgray'
        ))
        
        fig.add_trace(go.Bar(
            x=sender_stats['sender'],
            y=sender_stats['flagged_messages'],
            name='Flagged Messages',
            marker_color='crimson'
        ))
        
        fig.update_layout(
            barmode='stack',
            title="Messages by Sender",
            xaxis_title="Sender",
            yaxis_title="Message Count",
            height=400
        )
        
    else:
        # If no flagged messages, just show a simple bar chart
        fig = px.bar(
            sender_counts,
            x='sender',
            y='total_messages',
            title="Messages by Sender",
            color_discrete_sequence=['lightgray']
        )
        
        fig.update_layout(
            xaxis_title="Sender",
            yaxis_title="Message Count",
            height=400
        )
    
    st.plotly_chart(fig, use_container_width=True)

def create_sentiment_flow(conversation_data):
    """
    Create a visualization of message sentiment flow over time.
    
    Args:
        conversation_data: DataFrame containing all conversation messages
    """
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Add sentiment scores to the conversation data
    sentiment_df = conversation_data.copy()
    sentiment_df['sentiment'] = sentiment_df['message'].apply(
        lambda x: sia.polarity_scores(x)['compound']
    )
    
    # Create a line chart showing sentiment flow over time
    fig = px.line(
        sentiment_df,
        x='timestamp',
        y='sentiment',
        color='sender',
        hover_data=['message'],
        markers=True
    )
    
    # Add a horizontal line at sentiment = 0
    fig.add_shape(
        type="line",
        x0=sentiment_df['timestamp'].min(),
        y0=0,
        x1=sentiment_df['timestamp'].max(),
        y1=0,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    fig.update_layout(
        title="Message Sentiment Flow",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        height=400,
        yaxis=dict(
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_channel_distribution(flagged_df):
    """
    Create a visualization showing the distribution of flagged messages by channel.
    
    Args:
        flagged_df: DataFrame containing flagged messages
    """
    channel_counts = flagged_df['channel'].value_counts().reset_index()
    channel_counts.columns = ['channel', 'count']
    
    fig = px.pie(
        channel_counts,
        values='count',
        names='channel',
        title="Flagged Messages by Channel",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    fig.update_layout(
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_risk_keyword_heatmap(conversation_data):
    """
    Create a heatmap showing the occurrence of risk keywords over time.
    
    Args:
        conversation_data: DataFrame containing all conversation messages
    """
    # Define risk keywords categories for clarity
    risk_keywords = {
        'Technical Issues': ['spike', 'anomaly', 'thermal deviation', 'drift', 'failure', 'bug', 'error', 'sensor'],
        'Uncertainty': ['weird', 'unusual', 'unexpected', 'inconsistent', 'irregular', 'not sure', 'uncertain'],
        'Concerns': ['concern', 'issue', 'problem', 'not convinced', 'flag', 'warning', 'blockers']
    }
    
    # Create a time-binned version of the conversation
    # Use 5-minute bins for the time axis
    time_min = conversation_data['timestamp'].min()
    time_max = conversation_data['timestamp'].max()
    
    # Create time bins
    time_bins = pd.date_range(start=time_min, end=time_max, freq='5min')
    
    # Initialize the heatmap data
    heatmap_data = []
    
    for category, keywords in risk_keywords.items():
        for time_idx in range(len(time_bins) - 1):
            bin_start = time_bins[time_idx]
            bin_end = time_bins[time_idx + 1]
            
            # Messages in this time bin
            bin_messages = conversation_data[
                (conversation_data['timestamp'] >= bin_start) & 
                (conversation_data['timestamp'] < bin_end)
            ]
            
            # Count messages containing keywords in this category
            count = sum(1 for msg in bin_messages['message'] if any(keyword.lower() in msg.lower() for keyword in keywords))
            
            if len(bin_messages) > 0:
                normalized_count = count / len(bin_messages)
            else:
                normalized_count = 0
            
            heatmap_data.append({
                'time_bin': bin_start,
                'category': category,
                'count': normalized_count
            })
    
    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    
    if not heatmap_df.empty:
        # Pivot the DataFrame for the heatmap
        pivot_df = heatmap_df.pivot(index='category', columns='time_bin', values='count')
        
        # Create the heatmap
        fig = px.imshow(
            pivot_df.values,
            labels=dict(x="Time", y="Risk Category", color="Frequency"),
            x=[pd.to_datetime(str(col)).strftime('%H:%M') for col in pivot_df.columns],
            y=pivot_df.index,
            color_continuous_scale='Reds',
            aspect="auto"
        )
        
        fig.update_layout(
            title="Risk Keyword Frequency Over Time",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to generate a heatmap of risk keywords.")
