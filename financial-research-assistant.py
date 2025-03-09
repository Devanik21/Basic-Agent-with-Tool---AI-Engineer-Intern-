import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Dict, Any, List

# Enable seaborn for better visualizations
sns.set_theme(style="darkgrid")

# Define the state schema for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[Any], "The messages in the conversation so far"]
    ticker: Annotated[str, "The stock ticker symbol"]
    timeframe: Annotated[str, "The timeframe for analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"]
    stock_data: Annotated[Dict, "Raw stock data from yfinance"]
    analysis: Annotated[Dict, "Analysis results"]
    visualization_paths: Annotated[List[str], "Paths to generated visualization images"]
    recommendation: Annotated[str, "Investment recommendation"]

# Tool: Get stock data
def get_stock_data(ticker: str, timeframe: str) -> Dict:
    """
    Fetch stock data using yfinance API
    
    Args:
        ticker: Stock ticker symbol
        timeframe: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        Dictionary containing stock data and company info
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get stock info
        info = stock.info
        
        # Get historical data
        hist = stock.history(period=timeframe)
        
        # Get financial data
        financials = stock.financials.to_dict() if not stock.financials.empty else {}
        balance_sheet = stock.balance_sheet.to_dict() if not stock.balance_sheet.empty else {}
        cash_flow = stock.cashflow.to_dict() if not stock.cashflow.empty else {}
        
        # Calculate additional metrics
        if not hist.empty:
            # Calculate daily returns
            hist['Daily_Return'] = hist['Close'].pct_change()
            
            # Calculate volatility (standard deviation of returns)
            volatility = hist['Daily_Return'].std() * (252 ** 0.5)  # Annualized
            
            # Calculate moving averages
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
        
        return {
            'ticker': ticker,
            'company_name': info.get('shortName', 'Unknown'),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 'Unknown'),
            'pe_ratio': info.get('trailingPE', 'Unknown'),
            'dividend_yield': info.get('dividendYield', 'Unknown'),
            'historical_data': hist,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'Unknown')),
            'target_price': info.get('targetMeanPrice', 'Unknown'),
            'volatility': volatility if 'volatility' in locals() else 'Unknown'
        }
    except Exception as e:
        return {'error': str(e)}

# Tool: Analyze stock data
def analyze_stock(stock_data: Dict) -> Dict:
    """
    Analyze stock data and generate insights
    
    Args:
        stock_data: Dictionary containing stock data from yfinance
    
    Returns:
        Dictionary containing analysis results
    """
    analysis = {}
    
    if 'error' in stock_data:
        return {'error': stock_data['error']}
    
    try:
        # Basic company info
        analysis['company_overview'] = {
            'name': stock_data['company_name'],
            'sector': stock_data['sector'],
            'industry': stock_data['industry'],
            'market_cap': stock_data['market_cap']
        }
        
        # Price analysis
        hist_data = stock_data['historical_data']
        if not hist_data.empty:
            # Calculate performance metrics
            start_price = hist_data['Close'].iloc[0]
            end_price = hist_data['Close'].iloc[-1]
            price_change = end_price - start_price
            percent_change = (price_change / start_price) * 100
            
            analysis['performance'] = {
                'start_price': round(start_price, 2),
                'end_price': round(end_price, 2),
                'price_change': round(price_change, 2),
                'percent_change': round(percent_change, 2),
                'volatility': round(stock_data['volatility'], 4) if stock_data['volatility'] != 'Unknown' else 'Unknown'
            }
            
            # Technical indicators
            latest_price = hist_data['Close'].iloc[-1]
            ma20 = hist_data['MA20'].iloc[-1] if not np.isnan(hist_data['MA20'].iloc[-1]) else None
            ma50 = hist_data['MA50'].iloc[-1] if not np.isnan(hist_data['MA50'].iloc[-1]) else None
            ma200 = hist_data['MA200'].iloc[-1] if not np.isnan(hist_data['MA200'].iloc[-1]) else None
            
            analysis['technical_indicators'] = {
                'current_price': round(latest_price, 2),
                'MA20': round(ma20, 2) if ma20 is not None else None,
                'MA50': round(ma50, 2) if ma50 is not None else None,
                'MA200': round(ma200, 2) if ma200 is not None else None
            }
            
            # Trend analysis
            analysis['trend'] = {}
            if ma20 is not None and ma50 is not None:
                analysis['trend']['ma20_vs_ma50'] = 'Bullish' if ma20 > ma50 else 'Bearish'
            if ma50 is not None and ma200 is not None:
                analysis['trend']['ma50_vs_ma200'] = 'Bullish' if ma50 > ma200 else 'Bearish'
            
            # Calculate RSI (Relative Strength Index)
            delta = hist_data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            latest_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else None
            analysis['technical_indicators']['RSI'] = round(latest_rsi, 2) if latest_rsi is not None else None
            
        # Valuation metrics
        analysis['valuation'] = {
            'pe_ratio': stock_data['pe_ratio'],
            'dividend_yield': stock_data['dividend_yield'] if stock_data['dividend_yield'] != 'Unknown' else None,
            'target_price': stock_data['target_price']
        }
        
        return analysis
    except Exception as e:
        return {'error': str(e)}

# Tool: Generate visualizations
def generate_visualizations(ticker: str, stock_data: Dict, timeframe: str) -> List[str]:
    """
    Generate visualizations for the stock data
    
    Args:
        ticker: Stock ticker symbol
        stock_data: Dictionary containing stock data from yfinance
        timeframe: Time period for the analysis
    
    Returns:
        List of paths to the generated visualization images
    """
    visualization_paths = []
    
    if 'error' in stock_data:
        return visualization_paths
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        hist_data = stock_data['historical_data']
        if not hist_data.empty:
            # 1. Price Chart with Moving Averages
            plt.figure(figsize=(12, 6))
            plt.plot(hist_data.index, hist_data['Close'], label='Close Price', linewidth=2)
            
            # Add moving averages if available
            if 'MA20' in hist_data.columns:
                plt.plot(hist_data.index, hist_data['MA20'], label='20-day MA', linestyle='--')
            if 'MA50' in hist_data.columns:
                plt.plot(hist_data.index, hist_data['MA50'], label='50-day MA', linestyle='--')
            if 'MA200' in hist_data.columns:
                plt.plot(hist_data.index, hist_data['MA200'], label='200-day MA', linestyle='--')
            
            plt.title(f'{stock_data["company_name"]} ({ticker}) - Price Chart ({timeframe})')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            price_chart_path = f'visualizations/{ticker}_price_chart_{timeframe}.png'
            plt.savefig(price_chart_path)
            plt.close()
            visualization_paths.append(price_chart_path)
            
            # 2. Daily Returns Distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(hist_data['Daily_Return'].dropna(), kde=True, bins=30)
            plt.title(f'{stock_data["company_name"]} ({ticker}) - Daily Returns Distribution')
            plt.xlabel('Daily Return')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Save the figure
            returns_dist_path = f'visualizations/{ticker}_returns_distribution_{timeframe}.png'
            plt.savefig(returns_dist_path)
            plt.close()
            visualization_paths.append(returns_dist_path)
            
            # 3. Volume Chart
            plt.figure(figsize=(12, 6))
            plt.bar(hist_data.index, hist_data['Volume'], color='blue', alpha=0.7)
            plt.title(f'{stock_data["company_name"]} ({ticker}) - Trading Volume ({timeframe})')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.grid(True)
            
            # Save the figure
            volume_chart_path = f'visualizations/{ticker}_volume_{timeframe}.png'
            plt.savefig(volume_chart_path)
            plt.close()
            visualization_paths.append(volume_chart_path)
            
        return visualization_paths
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return visualization_paths

# Tool: Generate investment recommendation
def generate_recommendation(ticker: str, stock_data: Dict, analysis: Dict) -> str:
    """
    Generate investment recommendation based on analysis
    
    Args:
        ticker: Stock ticker symbol
        stock_data: Dictionary containing stock data from yfinance
        analysis: Dictionary containing analysis results
    
    Returns:
        Investment recommendation
    """
    if 'error' in stock_data or 'error' in analysis:
        return f"Unable to generate recommendation for {ticker} due to an error in the data or analysis."
    
    try:
        recommendation = f"Investment Recommendation for {analysis['company_overview']['name']} ({ticker}):\n\n"
        
        # Score different aspects (simple scoring system, can be enhanced)
        score = 0
        reasons = []
        
        # Technical analysis
        if 'technical_indicators' in analysis and 'trend' in analysis:
            # Moving average trends
            if 'ma20_vs_ma50' in analysis['trend']:
                if analysis['trend']['ma20_vs_ma50'] == 'Bullish':
                    score += 1
                    reasons.append("Short-term trend is bullish (20-day MA > 50-day MA)")
                else:
                    score -= 1
                    reasons.append("Short-term trend is bearish (20-day MA < 50-day MA)")
                    
            if 'ma50_vs_ma200' in analysis['trend']:
                if analysis['trend']['ma50_vs_ma200'] == 'Bullish':
                    score += 2
                    reasons.append("Long-term trend is bullish (50-day MA > 200-day MA)")
                else:
                    score -= 2
                    reasons.append("Long-term trend is bearish (50-day MA < 200-day MA)")
            
            # RSI analysis
            if 'RSI' in analysis['technical_indicators'] and analysis['technical_indicators']['RSI'] is not None:
                rsi = analysis['technical_indicators']['RSI']
                if rsi < 30:
                    score += 2
                    reasons.append(f"Stock is potentially oversold (RSI: {rsi})")
                elif rsi > 70:
                    score -= 2
                    reasons.append(f"Stock is potentially overbought (RSI: {rsi})")
        
        # Performance
        if 'performance' in analysis:
            perf = analysis['performance']
            if perf['percent_change'] > 10:
                score += 1
                reasons.append(f"Strong recent performance ({perf['percent_change']}% gain)")
            elif perf['percent_change'] < -10:
                score -= 1
                reasons.append(f"Poor recent performance ({perf['percent_change']}% loss)")
            
            # Volatility check
            if perf['volatility'] != 'Unknown':
                if perf['volatility'] > 0.4:  # High volatility
                    score -= 1
                    reasons.append(f"High volatility (Annual volatility: {perf['volatility']})")
        
        # Valuation metrics
        if 'valuation' in analysis:
            val = analysis['valuation']
            
            # P/E ratio analysis
            if val['pe_ratio'] != 'Unknown' and val['pe_ratio'] is not None:
                if 0 < val['pe_ratio'] < 15:
                    score += 2
                    reasons.append(f"Attractive valuation (P/E ratio: {val['pe_ratio']})")
                elif val['pe_ratio'] > 30:
                    score -= 1
                    reasons.append(f"Potentially overvalued (P/E ratio: {val['pe_ratio']})")
            
            # Dividend yield
            if val['dividend_yield'] not in ['Unknown', None]:
                if val['dividend_yield'] > 0.03:  # 3% dividend yield
                    score += 1
                    reasons.append(f"Good dividend yield ({val['dividend_yield']*100:.2f}%)")
            
            # Price vs target
            if val['target_price'] not in ['Unknown', None] and stock_data['current_price'] != 'Unknown':
                current = stock_data['current_price']
                target = val['target_price']
                
                if current < target:
                    upside = ((target - current) / current) * 100
                    if upside > 15:
                        score += 2
                        reasons.append(f"Significant upside potential ({upside:.2f}% to target price)")
                    elif upside > 5:
                        score += 1
                        reasons.append(f"Moderate upside potential ({upside:.2f}% to target price)")
                else:
                    downside = ((current - target) / current) * 100
                    if downside > 10:
                        score -= 2
                        reasons.append(f"Stock potentially overvalued ({downside:.2f}% above target price)")
        
        # Generate final recommendation
        if score >= 4:
            recommendation += "STRONG BUY\n\n"
        elif score >= 2:
            recommendation += "BUY\n\n"
        elif score >= 0:
            recommendation += "HOLD\n\n"
        elif score >= -2:
            recommendation += "SELL\n\n"
        else:
            recommendation += "STRONG SELL\n\n"
        
        # Add reasoning
        recommendation += "Key Factors:\n"
        for reason in reasons:
            recommendation += f"- {reason}\n"
        
        recommendation += "\nAdditional Considerations:\n"
        recommendation += f"- {analysis['company_overview']['name']} operates in the {analysis['company_overview']['sector']} sector, {analysis['company_overview']['industry']} industry.\n"
        
        # Add market cap context
        if analysis['company_overview']['market_cap'] != 'Unknown':
            market_cap = analysis['company_overview']['market_cap']
            if market_cap > 200_000_000_000:
                recommendation += f"- Large-cap stock with a market capitalization of ${market_cap/1_000_000_000:.2f} billion.\n"
            elif market_cap > 10_000_000_000:
                recommendation += f"- Mid-cap stock with a market capitalization of ${market_cap/1_000_000_000:.2f} billion.\n"
            elif market_cap > 2_000_000_000:
                recommendation += f"- Small-cap stock with a market capitalization of ${market_cap/1_000_000_000:.2f} billion.\n"
            else:
                recommendation += f"- Micro-cap stock with a market capitalization of ${market_cap/1_000_000_000:.2f} billion.\n"
        
        # Risk disclaimer
        recommendation += "\nDISCLAIMER: This is an automated recommendation based on technical and fundamental analysis. It should be considered alongside your own research and risk tolerance. Past performance does not guarantee future results."
        
        return recommendation
    except Exception as e:
        return f"Unable to generate recommendation for {ticker} due to an error: {str(e)}"

# Define the agent nodes
def data_collector_agent(state: AgentState) -> AgentState:
    """
    Agent responsible for collecting financial data
    """
    messages = state["messages"]
    ticker = state["ticker"]
    timeframe = state["timeframe"]
    
    # Fetch stock data using yfinance
    stock_data = get_stock_data(ticker, timeframe)
    
    # Update the state
    new_state = state.copy()
    new_state["stock_data"] = stock_data
    
    # Add agent message
    if "error" in stock_data:
        new_state["messages"] = messages + [AIMessage(content=f"Data Collector Agent: Failed to fetch data for {ticker}. Error: {stock_data['error']}")]
    else:
        new_state["messages"] = messages + [AIMessage(content=f"Data Collector Agent: Successfully fetched data for {ticker} over {timeframe} timeframe.")]
    
    return new_state

def analyst_agent(state: AgentState) -> AgentState:
    """
    Agent responsible for analyzing financial data
    """
    messages = state["messages"]
    ticker = state["ticker"]
    stock_data = state["stock_data"]
    
    # Perform analysis on the stock data
    analysis = analyze_stock(stock_data)
    
    # Update the state
    new_state = state.copy()
    new_state["analysis"] = analysis
    
    # Add agent message
    if "error" in analysis:
        new_state["messages"] = messages + [AIMessage(content=f"Analyst Agent: Failed to analyze data for {ticker}. Error: {analysis['error']}")]
    else:
        new_state["messages"] = messages + [AIMessage(content=f"Analyst Agent: Completed analysis for {ticker}.")]
    
    return new_state

def visualization_agent(state: AgentState) -> AgentState:
    """
    Agent responsible for generating visualizations
    """
    messages = state["messages"]
    ticker = state["ticker"]
    stock_data = state["stock_data"]
    timeframe = state["timeframe"]
    
    # Generate visualizations
    visualization_paths = generate_visualizations(ticker, stock_data, timeframe)
    
    # Update the state
    new_state = state.copy()
    new_state["visualization_paths"] = visualization_paths
    
    # Add agent message
    if not visualization_paths:
        new_state["messages"] = messages + [AIMessage(content=f"Visualization Agent: Failed to generate visualizations for {ticker}.")]
    else:
        vis_message = f"Visualization Agent: Generated {len(visualization_paths)} visualizations for {ticker}:\n"
        for path in visualization_paths:
            vis_message += f"- {path}\n"
        new_state["messages"] = messages + [AIMessage(content=vis_message)]
    
    return new_state

def advisor_agent(state: AgentState) -> AgentState:
    """
    Agent responsible for providing investment recommendations
    """
    messages = state["messages"]
    ticker = state["ticker"]
    stock_data = state["stock_data"]
    analysis = state["analysis"]
    
    # Generate investment recommendation
    recommendation = generate_recommendation(ticker, stock_data, analysis)
    
    # Update the state
    new_state = state.copy()
    new_state["recommendation"] = recommendation
    
    # Add agent message
    new_state["messages"] = messages + [AIMessage(content=f"Advisor Agent: Generated investment recommendation for {ticker}.")]
    
    return new_state

def should_continue(state: AgentState) -> str:
    """
    Determine if there was an error in any of the previous steps
    """
    stock_data = state.get("stock_data", {})
    analysis = state.get("analysis", {})
    
    if "error" in stock_data or "error" in analysis:
        return "end"
    return "continue"

# Define the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("data_collector", data_collector_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("visualization", visualization_agent)
workflow.add_node("advisor", advisor_agent)

# Add edges
workflow.add_edge("data_collector", should_continue)
workflow.add_conditional_edges(
    "data_collector",
    should_continue,
    {
        "continue": "analyst",
        "end": END
    }
)

workflow.add_edge("analyst", should_continue)
workflow.add_conditional_edges(
    "analyst",
    should_continue,
    {
        "continue": "visualization",
        "end": END
    }
)

workflow.add_edge("visualization", "advisor")
workflow.add_edge("advisor", END)

# Set the entry point
workflow.set_entry_point("data_collector")

# Compile the workflow
financial_research_assistant = workflow.compile()

# Function to run the financial research assistant
def run_financial_research(ticker: str, timeframe: str = "1y"):
    """
    Run the financial research assistant for a given ticker and timeframe
    
    Args:
        ticker: Stock ticker symbol
        timeframe: Time period for analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        Final state of the workflow
    """
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=f"Please analyze {ticker} stock over {timeframe} timeframe.")],
        "ticker": ticker,
        "timeframe": timeframe,
        "stock_data": {},
        "analysis": {},
        "visualization_paths": [],
        "recommendation": ""
    }
    
    # Run the workflow
    result = financial_research_assistant.invoke(initial_state)
    return result

# Example usage
if __name__ == "__main__":
    # Run analysis for Apple stock over a 1-year timeframe
    result = run_financial_research("AAPL", "1y")
    
    # Print the messages from each agent
    for message in result["messages"]:
        print(message.content)
    
    # Print the investment recommendation
    print("\nFINAL RECOMMENDATION:")
    print(result["recommendation"])
    
    # Display visualizations
    print("\nGenerated visualizations:")
    for path in result["visualization_paths"]:
        print(f"- {path}")
