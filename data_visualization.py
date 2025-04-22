import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

def generate_visualizations(conversation_data, flagged_messages, analyzed_conversation):
    """
    Generate visualizations for the risk detection results, focused on providing 
    clear business value for engineering leadership (CTO, CEO).
    
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
    
    # Critical executive visualizations focused on business value
    st.subheader("🚨 Critical Risk Indicators")
    
    # Create two columns for the key visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Timeline visualization of all messages with flagged messages highlighted
        create_timeline_visualization(conversation_data, flagged_df)
    
    with col2:
        # Sender distribution showing who is raising concerns vs dismissing them
        create_sender_distribution(conversation_data, flagged_df)
        
    # Add a severity score metric if we have flagged messages
    if not flagged_df.empty:
        create_executive_summary(flagged_df, conversation_data)

def create_timeline_visualization(all_messages, flagged_messages):
    """
    Create a timeline visualization showing all messages with flagged ones highlighted.
    
    Args:
        all_messages: DataFrame containing all conversation messages
        flagged_messages: DataFrame containing flagged messages
    """
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
    
    # Create a more compact timeline visualization
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top: 0;">Conversation Timeline</h3>', unsafe_allow_html=True)
    
    # Create the timeline visualization with more compact layout
    fig = px.scatter(
        timeline_df,
        x='timestamp',
        y='sender',
        color='flag_label',
        hover_data=['message', 'channel'],
        size_max=8,  # Smaller dot size
        color_discrete_map={
            'Normal Message': '#cccccc',  # Lighter gray for normal messages
            **{col: '#d13438' for col in timeline_df['flag_label'].unique() if col != 'Normal Message'}
        }
    )
    
    # Update layout to be more compact
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),  # Reduce margins
        xaxis_title=None,  # Remove axis titles for cleaner look
        yaxis_title=None,
        height=250,  # Reduce height
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='#eeeeee',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#eeeeee',
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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

def create_executive_summary(flagged_df, conversation_data):
    """
    Create an executive summary visualization focused on high-level risk metrics
    that would be valuable to a CTO or CEO.
    
    Args:
        flagged_df: DataFrame containing flagged messages
        conversation_data: DataFrame containing all conversation messages
    """
    # Calculate key metrics
    total_messages = len(conversation_data)
    total_flagged = len(flagged_df)
    flag_percentage = (total_flagged / total_messages * 100) if total_messages > 0 else 0
    
    # Calculate time span of conversation
    time_span = (conversation_data['timestamp'].max() - conversation_data['timestamp'].min()).total_seconds() / 60
    time_span = max(1, round(time_span))  # In minutes, minimum 1
    
    # Calculate dismissal ratio - number of messages with dismissal language
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment for leadership responses to concerns
    leadership_roles = ['PM_Lead', 'Director', 'QA_Tech', 'Systems_Admin']
    leadership_messages = conversation_data[conversation_data['sender'].isin(leadership_roles)]
    
    if not leadership_messages.empty:
        leadership_sentiments = leadership_messages['message'].apply(
            lambda x: sia.polarity_scores(x)['compound']
        ).tolist()
        avg_leadership_sentiment = np.mean(leadership_sentiments)
        dismissal_indicator = min(10, max(1, round((avg_leadership_sentiment + 1) * 5)))
    else:
        dismissal_indicator = 5  # Neutral if no leadership messages
    
    # Create risk severity score (1-10)
    if total_flagged == 0:
        risk_severity = 1
    else:
        # Calculate severity based on flagged percentage and dismissal indicator
        risk_severity = min(10, max(1, round((flag_percentage / 10) + dismissal_indicator)))
    
    # Determine risk level and color
    risk_level = "Low" if risk_severity <= 3 else "Medium" if risk_severity <= 7 else "High"
    risk_color = "#107c10" if risk_severity <= 3 else "#ff8c00" if risk_severity <= 7 else "#d13438"
    
    # Create compact metric cards in a container
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    
    # Compact header row with key risk indicators
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
        <h3>Key Risk Indicators</h3>
        <div style="background-color: {risk_color}22; padding: 0.3rem 0.8rem; border-radius: 4px;">
            <span style="font-weight: bold; color: {risk_color};">{risk_level} Risk Level</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a row of compact metric cards
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{risk_severity}/10</div>
            <div class="metric-label">Risk Severity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_flagged}</div>
            <div class="metric-label">Flagged Messages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{flag_percentage:.1f}%</div>
            <div class="metric-label">Flag Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        if not flagged_df.empty:
            first_flag_time = flagged_df['timestamp'].min()
            last_message_time = conversation_data['timestamp'].max()
            minutes_since_first_flag = (last_message_time - first_flag_time).total_seconds() / 60
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{minutes_since_first_flag:.0f}</div>
                <div class="metric-label">Minutes Since Flag</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{time_span:.0f}</div>
                <div class="metric-label">Minutes Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Add key people and channels in a second row if we have flagged messages
    if not flagged_df.empty:
        st.markdown("<hr style='margin: 0.8rem 0; border-color: #eee;'>", unsafe_allow_html=True)
        cols2 = st.columns(2)
        
        with cols2[0]:
            top_flagger = flagged_df['sender'].value_counts().index[0] if not flagged_df['sender'].value_counts().empty else "N/A"
            st.markdown(f"""
            <div style="padding: 0.5rem; background: #f8f9fa; border-radius: 4px;">
                <span style="font-weight: bold; font-size: 0.9rem;">Key Person Raising Concerns:</span>
                <div style="font-size: 1.1rem; margin-top: 0.2rem;">{top_flagger}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with cols2[1]:
            top_channel = flagged_df['channel'].mode().iloc[0] if not flagged_df['channel'].mode().empty else "N/A"
            st.markdown(f"""
            <div style="padding: 0.5rem; background: #f8f9fa; border-radius: 4px;">
                <span style="font-weight: bold; font-size: 0.9rem;">Most Active Risk Channel:</span>
                <div style="font-size: 1.1rem; margin-top: 0.2rem;">{top_channel}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create a timeline view of risk evolution
    if not flagged_df.empty:
        st.subheader("Risk Evolution Timeline")
        
        # Group flagged messages by 5-minute intervals for summary view
        flagged_df['time_group'] = flagged_df['timestamp'].dt.floor('5min')
        time_groups = flagged_df.groupby('time_group')
        
        # Create simplified timeline of risk evolution
        timeline_data = []
        for time, group in time_groups:
            timeline_data.append({
                'time': time,
                'count': len(group),
                'senders': ', '.join(group['sender'].unique()),
                'sample_message': group.iloc[0]['message']
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Create column chart for risk evolution
        if not timeline_df.empty:
            fig = px.bar(
                timeline_df, 
                x='time', 
                y='count',
                hover_data=['senders', 'sample_message'],
                labels={'count': 'Flagged Messages', 'time': 'Time'},
                color_discrete_sequence=['crimson']
            )
            
            fig.update_layout(
                title="Risk Flag Timeline",
                xaxis_title="Time",
                yaxis_title="Number of Flagged Messages",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
