
# Data Processing Flow

This document explains the complete process flow that occurs when a user uploads a CSV file to the Risk Detection System.

## Upload and Processing Overview

When a user uploads a CSV file, the system executes a series of operations to process the data, detect potential risks, and visualize the results. The following sections detail this flow.

## Step-by-Step Process Flow

### 1. Initial File Upload
- The upload process begins in `app.py` with the `st.file_uploader()` Streamlit component
- When a user selects a file, it's loaded into memory as an `UploadedFile` object
- The system verifies the file has a `.csv` extension

### 2. Data Validation and Preparation
- The CSV file is read into a pandas DataFrame using `pd.read_csv(uploaded_file)`
- The system validates that the required columns exist:
  - `timestamp`
  - `sender`
  - `channel`
  - `message`
- A preview of the data is displayed to the user
- When the user clicks "Analyze Conversation", the system:
  - Converts the `timestamp` column to datetime format using `pd.to_datetime()`
  - Stores the processed DataFrame in `st.session_state.conversation_data`

### 3. Risk Detection Process
- The main risk detection process begins in `app.py` by calling two core functions:
  - `detect_risks(df)` from `risk_detection.py`
  - `analyze_conversation(df)` from `risk_detection.py`

#### 3.1 Risk Detection Algorithm (`detect_risks` function)
1. Creates a copy of the input DataFrame to avoid modifying the original
2. Initializes NLTK's SentimentIntensityAnalyzer
3. Calculates sentiment scores for each message:
   - Applies `sia.polarity_scores()` to each message
   - Extracts and stores the compound sentiment score
4. Checks for risk keywords in each message:
   - Uses predefined `RISK_KEYWORDS` list to identify potential issues
   - Creates a boolean column `contains_risk_word`
5. Checks for dismissive language patterns:
   - Uses `DISMISSIVE_PATTERNS` list to identify language that downplays concerns
   - Creates a boolean column `is_dismissive`
6. Identifies leadership roles in the conversation:
   - Uses `LEADERSHIP_ROLES` list to tag messages from leaders
   - Creates a boolean column `is_leadership`
7. Flags messages that appear to be downplaying risks:
   - Combines risk keywords with dismissive language
   - Pays special attention when leadership is dismissing concerns
8. Identifies clusters of related messages using `find_risk_clusters(df_copy)`:
   - Uses CountVectorizer to convert messages to token counts
   - Calculates similarities between messages containing risk words
   - Groups related messages into clusters based on similarity threshold
9. Identifies dismissed concerns using `identify_dismissed_concerns(df_copy, risk_clusters)`:
   - For each cluster, analyzes the conversation flow
   - Flags messages where concerns are raised but later dismissed
   - Detects patterns of persistent concerns or doubt expressions
10. Returns a list of dictionaries containing flagged messages with reasons

#### 3.2 Conversation Analysis (`analyze_conversation` function)
1. Calculates overall statistics for the conversation:
   - Total messages
   - Messages per sender
   - Messages per channel
   - Average sentiment
   - Timeline statistics
2. Counts occurrences of risk keywords and dismissive language
3. Detects communication gaps using `detect_communication_gaps(df)`:
   - Groups messages into 5-minute time windows
   - Identifies windows where engineers raised concerns but leadership did not adequately respond
4. Returns a dictionary containing all analysis results

### 4. Results Storage
- The flagged messages are stored in `st.session_state.flagged_messages`
- The conversation analysis is stored in `st.session_state.analyzed_conversation`
- The application checks if any potential risks were detected:
  - If risks are found, `st.session_state.show_results` is set to `True`
  - If no risks are found, a message informs the user

### 5. Results Display
When the analysis is complete, the `display_results()` function in `app.py` is triggered:

1. Calculates risk metrics from the analysis results:
   - Flag percentage (percentage of messages that were flagged)
   - Determines risk level (High, Medium, Low) based on flag percentage
   - Assigns appropriate colors for visual indicators
2. Displays a risk assessment summary card with:
   - Risk level
   - Number of flagged messages
   - Flag rate percentage
3. Creates tabs for different views of the results:
   - Summary tab
   - Risk Details tab
   - Technical Data tab

#### 5.1 Summary Tab Content
1. Calls `generate_visualizations()` from `data_visualization.py` to create visual representations:
   - Creates timeline visualization with flagged messages highlighted
   - Displays sender distribution showing who is raising vs. dismissing concerns
   - Generates an executive summary with key risk indicators
   - Creates a risk evolution timeline
2. Calls `extract_insights()` from `message_analysis.py` to generate key insights:
   - Assesses risk severity
   - Identifies business impact
   - Analyzes key team members involved
   - Generates leadership action items

#### 5.2 Risk Details Tab Content
1. Provides a download option for the risk report as CSV
2. Calls `format_escalation_report()` from `utils.py` to generate a formatted report:
   - Groups flagged messages by reason
   - Creates collapsible sections for each reason category
   - Displays message details in a compact card format

#### 5.3 Technical Data Tab Content
- Displays raw data tables for detailed technical review
- Shows the complete DataFrame of flagged messages

## Key Functions and Data Flow

```
File Upload → app.py:main() → 
    → pd.read_csv() → Data Validation →
    → User clicks "Analyze" →
        → risk_detection.py:detect_risks() → 
            → find_risk_clusters() → 
            → identify_dismissed_concerns() →
        → risk_detection.py:analyze_conversation() →
            → detect_communication_gaps() →
    → app.py:display_results() →
        → data_visualization.py:generate_visualizations() →
            → create_timeline_visualization() →
            → create_sender_distribution() →
            → create_executive_summary() →
        → message_analysis.py:extract_insights() →
            → assess_risk_severity() →
            → identify_risk_themes() →
            → generate_recommendations() →
        → utils.py:format_escalation_report()
```

## Data Transformation

Throughout the process, the raw CSV data undergoes several transformations:

1. **Raw CSV** → **pandas DataFrame** (Initial loading)
2. **DataFrame** → **Enriched DataFrame** (Sentiment analysis, risk keywords, etc.)
3. **Enriched DataFrame** → **Risk Clusters** (Related messages grouped)
4. **Risk Clusters** → **Flagged Messages List** (Specific messages requiring attention)
5. **Flagged Messages List** → **Visualizations & Reports** (Visual presentation of results)

## Behind the Scenes: NLP Techniques

The risk detection process employs several Natural Language Processing techniques:

1. **Sentiment Analysis**: Uses NLTK's SentimentIntensityAnalyzer to determine message tone
2. **Keyword Matching**: Searches for predefined risk and dismissive terms
3. **Text Vectorization**: Converts messages to numerical vectors for similarity comparison
4. **Cosine Similarity**: Measures text similarity to identify related messages
5. **Pattern Recognition**: Identifies patterns of expressed concerns and dismissals

## Performance Considerations

For large CSV files, the system:

1. Creates a copy of the DataFrame to preserve original data
2. Uses efficient vectorized operations where possible
3. Employs a spinner to indicate processing is underway
4. Stores results in session state to avoid reprocessing

## Error Handling

The upload process includes error handling for:

1. Invalid file formats (non-CSV files)
2. Missing required columns
3. Processing errors (with detailed error messages)
4. Empty result sets (informing the user when no risks are detected)
