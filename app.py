import streamlit as st
import pandas as pd
import io
import time
import sys
import os

# Run setup to download required NLTK data
try:
    import setup
except Exception as e:
    st.error(f"Failed to run setup: {str(e)}")

from risk_detection import analyze_conversation, detect_risks
from data_visualization import generate_visualizations
from message_analysis import extract_insights
from utils import format_escalation_report

st.set_page_config(
    page_title="Risk Detection System",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Engineering Risk Detection System")
    st.markdown("""
    This tool analyzes Slack conversations to identify potential risks that might be buried or downplayed.
    Upload a CSV file of your Slack messages and get an escalation report with visualizations.
    """)
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your Slack conversation CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if the required columns exist
            required_columns = ["timestamp", "sender", "channel", "message"]
            if not all(col in df.columns for col in required_columns):
                st.error("CSV must contain the following columns: timestamp, sender, channel, message")
                return
            
            # Display data preview
            with st.expander("Preview of uploaded data"):
                st.dataframe(df.head(10))
            
            # Process the data when user clicks the button
            if st.button("Analyze Conversation"):
                with st.spinner("Analyzing conversation for buried risks..."):
                    # Add a small delay to show the spinner
                    time.sleep(1)
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Process the conversation and detect risks
                    st.session_state.conversation_data = df
                    st.session_state.flagged_messages = detect_risks(df)
                    st.session_state.analyzed_conversation = analyze_conversation(df)
                    
                    # If we have results, set the flag to display them
                    if len(st.session_state.flagged_messages) > 0:
                        st.session_state.show_results = True
                    else:
                        st.session_state.show_results = False
                        st.info("No potential risks were detected in this conversation.")
                
                st.success("Analysis complete!")
                
                # Rerun to display results
                st.rerun()
        
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
    
    # Display results if available
    if 'show_results' in st.session_state and st.session_state.show_results:
        display_results()

def display_results():
    st.header("📊 Analysis Results")
    
    # Get data from session state
    flagged_messages = st.session_state.flagged_messages
    conversation_data = st.session_state.conversation_data
    analyzed_conversation = st.session_state.analyzed_conversation
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Escalation Report", "Visualizations", "Insights"])
    
    with tab1:
        st.subheader("🚨 Escalation Report")
        if len(flagged_messages) > 0:
            # Format and display the escalation report
            escalation_report = format_escalation_report(flagged_messages)
            st.markdown(escalation_report)
            
            # Download option for the report
            csv_buffer = io.StringIO()
            pd.DataFrame(flagged_messages).to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Escalation Report as CSV",
                data=csv_buffer.getvalue(),
                file_name="escalation_report.csv",
                mime="text/csv"
            )
        else:
            st.info("No messages were flagged for escalation.")
    
    with tab2:
        st.subheader("📈 Risk Visualizations")
        # Generate and display visualizations
        generate_visualizations(conversation_data, flagged_messages, analyzed_conversation)
    
    with tab3:
        st.subheader("🔍 Key Insights")
        # Extract and display insights from the conversation
        insights = extract_insights(conversation_data, flagged_messages, analyzed_conversation)
        st.markdown(insights)

if __name__ == "__main__":
    # Initialize session state variables if they don't exist
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'flagged_messages' not in st.session_state:
        st.session_state.flagged_messages = []
    if 'conversation_data' not in st.session_state:
        st.session_state.conversation_data = None
    if 'analyzed_conversation' not in st.session_state:
        st.session_state.analyzed_conversation = None
        
    main()
