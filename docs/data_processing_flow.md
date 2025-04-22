
# Data Processing Flow Documentation

This document outlines the complete process that occurs when a CSV file is uploaded to the Risk Detection System.

## Upload and Processing Workflow

When you upload a CSV file in the Risk Detection System, the following sequence of operations occurs:

### 1. File Upload (app.py)

The process begins in the `app.py` file which contains the main Streamlit application:

```python
uploaded_file = st.file_uploader("Upload your Slack conversation CSV file", type=["csv"])
```

When a file is uploaded:
1. Streamlit handles the file upload and stores it temporarily
2. The system waits for the user to click the "Analyze Conversation" button
3. No processing occurs until user confirmation to analyze

### 2. Initial Data Validation (app.py)

Once the user clicks "Analyze Conversation", the system performs initial validation:

```python
if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Check if the required columns exist
        required_columns = ["timestamp", "sender", "channel", "message"]
        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain the following columns: timestamp, sender, channel, message")
            return
```

This ensures that:
- The uploaded file can be parsed as a valid CSV
- All required columns are present (timestamp, sender, channel, message)

### 3. Data Preprocessing (app.py)

After validation, the following preprocessing steps occur:

```python
# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
```

This converts string timestamps to Python datetime objects for better analysis and visualization.

### 4. Risk Detection (risk_detection.py)

The `detect_risks` function is the primary analysis function:

```python
st.session_state.flagged_messages = detect_risks(df)
```

This function performs several operations:
1. **Sentiment Analysis**: Computes sentiment scores for each message
   ```python
   sia = SentimentIntensityAnalyzer()
   df_copy['sentiment'] = df_copy['message'].apply(lambda x: sia.polarity_scores(x))
   df_copy['compound_sentiment'] = df_copy['sentiment'].apply(lambda x: x['compound'])
   ```

2. **Risk Keyword Detection**: Identifies messages containing known risk-related terms
   ```python
   df_copy['contains_risk_word'] = df_copy['message'].apply(
       lambda x: any(keyword.lower() in x.lower() for keyword in RISK_KEYWORDS)
   )
   ```

3. **Dismissive Language Detection**: Detects language patterns that might be downplaying concerns
   ```python
   df_copy['is_dismissive'] = df_copy['message'].apply(
       lambda x: any(re.search(r'\b' + re.escape(pattern.lower()) + r'\b', x.lower()) for pattern in DISMISSIVE_PATTERNS)
   )
   ```

4. **Role Analysis**: Considers the sender's role in the conversation
   ```python
   df_copy['is_leadership'] = df_copy['sender'].isin(LEADERSHIP_ROLES)
   ```

5. **Risk Clustering**: Groups related risk messages using NLP techniques
   ```python
   risk_clusters = find_risk_clusters(df_copy)
   ```

6. **Concern Identification**: Finds messages raising concerns that may have been dismissed
   ```python
   flagged_messages = identify_dismissed_concerns(df_copy, risk_clusters)
   ```

### 5. Conversation Analysis (app.py)

In parallel with risk detection, a broader conversation analysis occurs:

```python
st.session_state.analyzed_conversation = analyze_conversation(df)
```

This function extracts patterns and statistics:
1. Overall sentiment trends
2. Message distribution across senders and channels
3. Communication gaps and potential disconnects
4. Timeline of the conversation

### 6. Results Storage (app.py)

The results are stored in the Streamlit session state for display:

```python
st.session_state.conversation_data = df
st.session_state.flagged_messages = detect_risks(df)
st.session_state.analyzed_conversation = analyze_conversation(df)
```

### 7. Results Display (app.py)

Finally, the results are displayed through the `display_results` function:

```python
def display_results():
    st.header("📊 Analysis Results")
    
    # Get data from session state
    flagged_messages = st.session_state.flagged_messages
    conversation_data = st.session_state.conversation_data
    analyzed_conversation = st.session_state.analyzed_conversation
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Escalation Report", "Visualizations", "Insights"])
```

This displays the results in three tabs:

#### a. Escalation Report (utils.py)
The `format_escalation_report` function formats the flagged messages into a readable report:

```python
escalation_report = format_escalation_report(flagged_messages)
```

This report includes:
- Timestamps and senders of flagged messages
- The actual message content
- Reasons why each message was flagged
- A summary of themes and patterns

#### b. Visualizations (data_visualization.py)
The `generate_visualizations` function creates interactive visualizations:

```python
generate_visualizations(conversation_data, flagged_messages, analyzed_conversation)
```

This includes:
- Timeline of the conversation with flagged messages highlighted
- Message distribution by sender
- Executive summary with risk severity metrics
- Risk evolution timeline

#### c. Insights (message_analysis.py)
The `extract_insights` function provides a deeper analysis:

```python
insights = extract_insights(conversation_data, flagged_messages, analyzed_conversation)
```

This includes:
- Key risk themes identified
- Communication pattern insights
- Risk severity assessment
- Recommendations based on findings

## Data Flow Diagram

```
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│  CSV Upload │────▶│ Data Validation│────▶│ Preprocessing  │
└─────────────┘     └───────────────┘     └────────────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│  Insights   │◀────│ Visualizations│◀────│ Risk Detection │
└─────────────┘     └───────────────┘     └────────────────┘
       ▲                    ▲                     │
       │                    │                     │
       └────────────────────┴─────────────────────┘
                           │
                           ▼
                    ┌────────────────┐
                    │ Results Display│
                    └────────────────┘
```

## Detailed Function Execution Order

1. **Initial App Loading**:
   - `main()` in app.py
   - File uploader component initialization

2. **Upon File Upload**:
   - CSV validation
   - Data preview display

3. **Upon Analysis Button Click**:
   - Timestamp conversion
   - `detect_risks(df)` → Returns flagged messages
   - `analyze_conversation(df)` → Returns analysis metrics
   - Store results in session state

4. **Results Display**:
   - `display_results()` function call
   - `format_escalation_report()` → Formats the report
   - `generate_visualizations()` → Creates charts
   - `extract_insights()` → Provides analysis

## Key Algorithms in Processing

### Risk Detection Algorithm

```
1. Apply sentiment analysis to each message
2. Identify messages containing risk keywords
3. Detect dismissive language patterns
4. Check if senders are in leadership roles
5. Identify potentially downplayed risks using these factors
6. Cluster related risk messages
7. Flag messages that raise concerns but are dismissed
8. Sort flagged messages by timestamp
```

### Message Clustering Algorithm

```
1. Extract messages containing risk keywords
2. Create a corpus of just these messages
3. Convert to token count matrix using CountVectorizer
4. Normalize the vectors
5. Calculate cosine similarity between messages
6. Group messages that exceed similarity threshold
7. Return clusters of related messages
```

### Visualization Generation

```
1. Create timeline visualization with flagged messages highlighted
2. Generate message distribution by sender
3. If flagged messages exist, create executive summary
4. Calculate risk severity score
5. Display metrics with appropriate color coding
6. If applicable, create risk evolution timeline
```

## File Dependencies

- **app.py**: Main application logic and UI
- **risk_detection.py**: Core risk detection algorithms
- **message_analysis.py**: Analysis of conversation patterns
- **data_visualization.py**: Visualization creation
- **utils.py**: Helper functions for formatting

This comprehensive data flow ensures that all uploaded CSV files are processed systematically to extract maximum value from the conversation data.
