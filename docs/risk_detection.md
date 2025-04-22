
# Risk Detection Algorithms

## Key Components

The Risk Detection System uses several algorithmic approaches to identify potential risks in engineering conversations:

### 1. Keyword-Based Detection

The system maintains a list of risk-related keywords that might indicate technical problems:

```python
RISK_KEYWORDS = [
    'spike', 'anomaly', 'weird', 'thermal deviation', 'not urgent but', 
    'deviation', 'unusual', 'abnormal', 'drift', 'fluctuation', 'issue',
    'bug', 'glitch', 'error', 'warning', 'concern', 'problem', 'malfunction',
    'failure', 'fault', 'defect', 'inconsistent', 'unexpected', 'irregular'
]
```

### 2. Dismissive Language Detection

The system identifies phrases that might be used to downplay concerns:

```python
DISMISSIVE_PATTERNS = [
    'not a big deal', 'probably nothing', 'don\'t worry', 'not critical',
    'no need to', 'minor', 'not urgent', 'can ignore', 'non-blocking',
    'not a showstopper', 'not alarming', 'within tolerance', 'noise',
    'harmless', 'nothing to worry about', 'not a concern', 'deemed non-blocking',
    'no criticals', 'not prioritize', 'all clear', 'no red flags'
]
```

### 3. Sentiment Analysis

The system analyzes the sentiment of each message using NLTK's SentimentIntensityAnalyzer:

```python
sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(message)
compound_score = sentiment['compound']
```

### 4. Role-Based Analysis

The system considers the sender's role when evaluating messages, with special attention to leadership roles:

```python
LEADERSHIP_ROLES = ['PM_Lead', 'Director', 'QA_Tech', 'Systems_Admin']
```

## Detection Logic

The core detection logic combines these components to flag messages that might indicate buried risks:

```python
downplaying_risk = (
    (contains_risk_word & is_dismissive) | 
    (contains_risk_word & (compound_sentiment > 0) & is_leadership)
)
```

This flags messages that either:
1. Contain risk keywords and dismissive language, or
2. Contain risk keywords with positive sentiment from leadership roles

## Additional Detection Methods

### Communication Gap Detection

The system identifies potential gaps in communication where concerns might be ignored:

1. Message sequences where technical concerns are followed by dismissive responses
2. Repeated mentions of the same concern without acknowledgment
3. Changes in conversation topic without addressing raised concerns

### Temporal Pattern Analysis

The system analyzes message timing to identify:

1. Delayed responses to concerns
2. Rapid changes in conversation topic after concerns are raised
3. Clusters of risk-related messages that might indicate escalating problems

## Risk Severity Assessment

The system assesses the severity of detected risks based on:

1. Dismissal factor: How often concerns were dismissed
2. Persistence factor: Whether concerns were raised multiple times
3. Impact potential: Presence of words indicating serious consequences

## Continuous Improvement

The risk detection algorithms are designed to be expanded with:

1. Additional domain-specific keywords for different engineering disciplines
2. Enhanced pattern recognition for complex risk indicators
3. Machine learning capabilities to improve detection accuracy over time
