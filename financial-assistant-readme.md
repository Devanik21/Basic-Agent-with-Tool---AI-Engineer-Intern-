# Financial Research Assistant

A multi-agent system that analyzes stocks, visualizes financial data, and provides investment recommendations.

## Overview

The Financial Research Assistant is an intelligent system built using LangGraph that automates the stock analysis process. It combines financial data retrieval, technical analysis, visualization, and investment recommendations in a seamless workflow powered by multiple specialized agents.

## System Capabilities

The Financial Research Assistant is capable of:

1. **Data Collection**:
   - Fetches comprehensive financial data for any stock using yfinance
   - Gets historical prices, company information, financial statements
   - Calculates technical indicators like moving averages

2. **Financial Analysis**:
   - Performs technical analysis (moving averages, RSI)
   - Calculates performance metrics (price changes, volatility)
   - Analyzes valuation metrics (P/E ratio, dividend yield)

3. **Visualization**:
   - Creates price charts with moving averages
   - Generates daily returns distribution visualization
   - Displays trading volume over time

4. **Investment Advising**:
   - Provides actionable investment recommendations (Buy, Hold, Sell)
   - Explains reasoning for recommendations
   - Includes risk disclaimers and context

## How It Works

The system utilizes a LangGraph workflow with four specialized agents:

1. **Data Collector Agent**: Fetches financial data using yfinance
   - Retrieves historical prices, company info, and financial statements
   - Calculates technical indicators (moving averages, volatility)

2. **Analyst Agent**: Analyzes financial metrics and historical data
   - Performs technical analysis (trend identification, RSI calculation)
   - Evaluates performance metrics and valuation indicators

3. **Visualization Agent**: Creates charts and graphs of the data
   - Generates price charts with moving averages
   - Creates distributions of daily returns
   - Visualizes trading volume

4. **Advisor Agent**: Provides investment recommendations
   - Uses a scoring system based on technical and fundamental factors
   - Generates Buy/Hold/Sell recommendations with explanations
   - Includes contextual information about the company

The agents communicate through a shared state that maintains the analysis context. If any agent encounters an error, the workflow stops gracefully, preventing incomplete or inaccurate analysis.

## Key Features

- **Custom yfinance Tool**: The system leverages a custom tool built around the yfinance API to fetch comprehensive financial data
- **Multi-Agent Architecture**: Different specialized agents handle specific parts of the analysis process
- **Visualization Generation**: Creates professional-grade visualizations for better decision making
- **Investment Scoring System**: Uses a weighted scoring system to generate recommendations based on multiple factors
- **Error Handling**: Robust error handling prevents the system from crashing if data is unavailable
- **Conditional Workflow**: The graph structure ensures that analysis only proceeds when valid data is available

## Usage Example

To use this system:

```python
# Import the system
from financial_research_assistant import run_financial_research

# Analyze Tesla stock over a 6-month period
result = run_financial_research("TSLA", "6mo")

# Get the recommendation
print(result["recommendation"])

# View agent conversation
for message in result["messages"]:
    print(message.content)

# Access generated visualizations
print(result["visualization_paths"])
```

## Timeframe Options

The system supports various timeframes for analysis:
- `1d`: 1 day
- `5d`: 5 days
- `1mo`: 1 month
- `3mo`: 3 months
- `6mo`: 6 months
- `1y`: 1 year
- `2y`: 2 years
- `5y`: 5 years
- `10y`: 10 years
- `ytd`: Year to date
- `max`: Maximum available data

## Installation

1. Clone the repository
2. Install the required dependencies:
```
pip install yfinance pandas matplotlib seaborn langgraph langchain-core
```

## Requirements

- Python 3.8+
- yfinance
- pandas
- matplotlib
- seaborn
- langgraph
- langchain-core

## Disclaimer

This tool is for informational purposes only. The investment recommendations should be considered alongside your own research and risk tolerance. Past performance does not guarantee future results.
