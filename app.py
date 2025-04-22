import streamlit as st
import pandas as pd
import io
import time

from risk_detection import analyze_conversation, detect_risks
from data_visualization import generate_visualizations
from message_analysis import extract_insights
from utils import format_escalation_report

st.set_page_config(
    page_title="Risk Detection System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
    }
    .css-1v3fvcr {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #0078d4;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #106ebe;
    }
    .stDownloadButton>button {
        background-color: #107c10;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stDownloadButton>button:hover {
        background-color: #0b5c0b;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Professional header with corporate-style design
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <h1 style="margin: 0; color: #0078d4;">Engineering Risk Detection System</h1>
        <p style="margin: 0; color: #666; font-size: 0.9rem;">Powered by AI Analytics</p>
    </div>
    <hr style="margin-top: 0; margin-bottom: 2rem; border-color: #ddd;">
    """, unsafe_allow_html=True)
    
    # Executive-friendly introduction
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 2rem;">
        <h3 style="margin-top: 0;">Executive Summary</h3>
        <p>This system identifies potentially critical engineering risks that may be buried or downplayed in team communications. 
        Upload a CSV export of Slack conversations to receive an analysis of overlooked issues that may require leadership attention.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    # Get data from session state
    flagged_messages = st.session_state.flagged_messages
    conversation_data = st.session_state.conversation_data
    analyzed_conversation = st.session_state.analyzed_conversation
    
    # Calculate severity for executive header
    total_messages = len(conversation_data)
    total_flagged = len(flagged_messages) if flagged_messages else 0
    flag_percentage = (total_flagged / total_messages * 100) if total_messages > 0 else 0
    
    # Determine risk level
    if flag_percentage > 15:
        risk_level = "High"
        risk_color = "#d13438"
    elif flag_percentage > 5:
        risk_level = "Medium"
        risk_color = "#ff8c00"
    else:
        risk_level = "Low" if total_flagged > 0 else "None"
        risk_color = "#107c10" if total_flagged > 0 else "#666666"
    
    # Executive header with risk assessment
    st.markdown(f"""
    <div style="background-color: {risk_color}22; padding: 1.5rem; border-left: 5px solid {risk_color}; 
    border-radius: 4px; margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin: 0; color: {risk_color};">Risk Assessment: {risk_level}</h2>
            <p style="margin: 0.5rem 0 0 0;">Analysis identified {total_flagged} potentially concerning messages that may require leadership attention.</p>
        </div>
        <div style="text-align: right;">
            <h3 style="margin: 0; color: {risk_color};">{flag_percentage:.1f}%</h3>
            <p style="margin: 0; font-size: 0.9rem;">Flagged Rate</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections with executive-friendly labels
    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Risk Metrics", "Detailed Analysis"])
    
    with tab1:
        # Generate and display visualizations first - these are now more executive-friendly
        generate_visualizations(conversation_data, flagged_messages, analyzed_conversation)
        
        # Then show key insights below the visualizations
        st.subheader("🔍 Key Business Insights")
        insights = extract_insights(conversation_data, flagged_messages, analyzed_conversation)
        st.markdown(insights)
    
    with tab2:
        # Create a more structured escalation report with better formatting
        if len(flagged_messages) > 0:
            # Actions row with download buttons
            col1, col2 = st.columns([3, 1])
            with col2:
                # Download option for the report
                csv_buffer = io.StringIO()
                pd.DataFrame(flagged_messages).to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Export Risk Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="risk_escalation_report.csv",
                    mime="text/csv"
                )
            
            # Format and display the escalation report
            escalation_report = format_escalation_report(flagged_messages)
            st.markdown(escalation_report)
        else:
            st.info("No messages were flagged for escalation.")
    
    with tab3:
        # More detailed technical information for those who want to dig deeper
        if len(flagged_messages) > 0:
            st.subheader("Technical Analysis Details")
            st.markdown("""
            This section provides more detailed information about the analysis results for technical team members.
            """)
            
            # Display raw data tables for detailed review
            st.subheader("Raw Flagged Messages Data")
            st.dataframe(pd.DataFrame(flagged_messages), use_container_width=True)

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
