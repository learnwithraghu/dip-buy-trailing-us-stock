import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DJIA Stock Investment Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TOTAL_CAPITAL_AED = 50000
ALLOCATION_PER_STOCK_AED = 1250
MAX_STOCKS_IF_TOP_DROP_GT_10 = 3
MAX_STOCKS_IF_TOP_DROP_LE_10 = 1
DROP_THRESHOLD_PERCENT = -10
TARGET_PERCENTAGE = 3.14  # Target profit percentage

# DJIA Stock Symbols (30 stocks)
DJIA_SYMBOLS = [
    'AAPL', 'MSFT', 'JPM', 'V', 'PG', 'UNH', 'HD', 'DIS', 'VZ', 'CVX',
    'KO', 'MRK', 'PFE', 'INTC', 'IBM', 'XOM', 'GS', 'MCD', 'AXP', 'BA',
    'CAT', 'CSCO', 'JNJ', 'MMM', 'NKE', 'TRV', 'WBA', 'WMT', 'DOW', 'AMGN'
]

# USD to AED exchange rate (approximate, can be updated)
USD_TO_AED = 3.67

# Tracker CSV file path
TRACKER_CSV_FILE = "stock_tracker.csv"


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol):
    """Fetch stock data for a given symbol"""
    try:
        logger.info(f"Fetching data for {symbol}...")
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            logger.warning(f"No historical data found for {symbol}")
            return None
        
        logger.info(f"Retrieved {len(hist)} days of data for {symbol}")
        info = ticker.info
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        high_52w = hist['High'].max()
        
        drop_pct = ((high_52w - current_price) / high_52w) * 100 if high_52w > 0 else 0
        price_change = current_price - prev_close
        
        logger.info(f"{symbol}: Price=${current_price:.2f}, 52W High=${high_52w:.2f}, Drop={drop_pct:.2f}%, Change=${price_change:.2f}")
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'prev_close': prev_close,
            'high_52w': high_52w,
            'price_change': price_change,
            'price_change_pct': ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0,
            'drop_from_52w': drop_pct,
            'name': info.get('longName', symbol)
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}", exc_info=True)
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None


def analyze_stocks():
    """Fetch and analyze all DJIA stocks"""
    logger.info("=" * 50)
    logger.info("Starting stock analysis...")
    logger.info(f"Total stocks to analyze: {len(DJIA_SYMBOLS)}")
    
    data_list = []
    
    with st.spinner("Fetching stock data..."):
        progress_bar = st.progress(0)
        total_stocks = len(DJIA_SYMBOLS)
        
        for i, symbol in enumerate(DJIA_SYMBOLS):
            logger.info(f"Processing {i+1}/{total_stocks}: {symbol}")
            data = fetch_stock_data(symbol)
            if data:
                data_list.append(data)
                logger.info(f"✓ Successfully fetched {symbol}")
            else:
                logger.warning(f"✗ Failed to fetch {symbol}")
            progress_bar.progress((i + 1) / total_stocks)
            time.sleep(0.1)  # Small delay to avoid rate limiting
        
        progress_bar.empty()
    
    logger.info(f"Successfully fetched data for {len(data_list)} out of {total_stocks} stocks")
    
    if not data_list:
        logger.error("No stock data available!")
        st.error("No stock data available. Please try again later.")
        return None, None
    
    # Create DataFrame
    logger.info("Creating DataFrame...")
    df = pd.DataFrame(data_list)
    logger.info(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    
    # Filter stocks meeting conditions:
    # A) Fallen from 52-week high
    # B) Trading above last day's closing price
    logger.info("Applying filter conditions...")
    df['meets_condition_a'] = df['drop_from_52w'] > 0  # Has fallen from 52W high
    df['meets_condition_b'] = df['price_change'] > 0  # Rising from last day's close
    df['meets_both_conditions'] = df['meets_condition_a'] & df['meets_condition_b']
    
    condition_a_count = df['meets_condition_a'].sum()
    condition_b_count = df['meets_condition_b'].sum()
    both_conditions_count = df['meets_both_conditions'].sum()
    
    logger.info(f"Condition A (Dropped from 52W): {condition_a_count} stocks")
    logger.info(f"Condition B (Rising today): {condition_b_count} stocks")
    logger.info(f"Both conditions met: {both_conditions_count} stocks")
    
    # Sort by drop from 52W high (descending - most dropped first)
    df = df.sort_values('drop_from_52w', ascending=False, ignore_index=True)
    logger.info("DataFrame sorted by drop from 52W high")
    
    return df, data_list


def determine_buy_stocks(df):
    """Determine which stocks to buy based on the algorithm"""
    logger.info("=" * 50)
    logger.info("Determining buy stocks...")
    
    # Filter stocks that meet both conditions
    eligible_stocks = df[df['meets_both_conditions']].copy()
    logger.info(f"Eligible stocks (meet both conditions): {len(eligible_stocks)}")
    
    if eligible_stocks.empty:
        logger.info("No eligible stocks found - returning empty DataFrame")
        return pd.DataFrame()
    
    # Get the top stock's drop percentage
    top_stock = eligible_stocks.iloc[0]
    top_stock_drop = top_stock['drop_from_52w']
    top_stock_symbol = top_stock['symbol']
    
    logger.info(f"Top stock: {top_stock_symbol} with {top_stock_drop:.2f}% drop from 52W high")
    logger.info(f"Threshold: {abs(DROP_THRESHOLD_PERCENT)}%")
    
    # Determine how many stocks to buy
    if top_stock_drop > abs(DROP_THRESHOLD_PERCENT):  # Fallen more than -10%
        # Buy max 3 stocks
        max_stocks = MAX_STOCKS_IF_TOP_DROP_GT_10
        logger.info(f"Top stock dropped > {abs(DROP_THRESHOLD_PERCENT)}% - buying up to {max_stocks} stocks")
        buy_stocks = eligible_stocks.head(max_stocks)
    else:  # Fallen less than or equal to -10%
        # Buy only 1 stock
        max_stocks = MAX_STOCKS_IF_TOP_DROP_LE_10
        logger.info(f"Top stock dropped <= {abs(DROP_THRESHOLD_PERCENT)}% - buying only {max_stocks} stock")
        buy_stocks = eligible_stocks.head(max_stocks)
    
    logger.info(f"Selected {len(buy_stocks)} stock(s) for purchase:")
    for idx, stock in buy_stocks.iterrows():
        logger.info(f"  - {stock['symbol']}: {stock['name']} (Drop: {stock['drop_from_52w']:.2f}%)")
    
    # Calculate number of shares to buy for each stock
    buy_stocks = buy_stocks.copy()
    buy_stocks['allocation_aed'] = ALLOCATION_PER_STOCK_AED
    buy_stocks['allocation_usd'] = ALLOCATION_PER_STOCK_AED / USD_TO_AED
    buy_stocks['shares_to_buy'] = (buy_stocks['allocation_usd'] / buy_stocks['current_price']).apply(lambda x: int(x) if not pd.isna(x) else 0)
    buy_stocks['total_cost_usd'] = buy_stocks['shares_to_buy'] * buy_stocks['current_price']
    buy_stocks['total_cost_aed'] = buy_stocks['total_cost_usd'] * USD_TO_AED
    
    logger.info("Calculated share allocations:")
    for idx, stock in buy_stocks.iterrows():
        logger.info(f"  - {stock['symbol']}: {stock['shares_to_buy']} shares @ ${stock['current_price']:.2f} = ${stock['total_cost_usd']:.2f} USD")
    
    return buy_stocks


def format_currency(value, currency="USD"):
    """Format currency value"""
    if currency == "USD":
        return f"${value:,.2f}"
    elif currency == "AED":
        return f"AED {value:,.2f}"
    return f"{value:,.2f}"


def load_tracker_from_csv():
    """Load tracked stocks from CSV file"""
    if os.path.exists(TRACKER_CSV_FILE):
        try:
            df = pd.read_csv(TRACKER_CSV_FILE)
            tracked_stocks = df.to_dict('records')
            logger.info(f"Loaded {len(tracked_stocks)} stocks from {TRACKER_CSV_FILE}")
            return tracked_stocks
        except Exception as e:
            logger.error(f"Error loading tracker CSV: {str(e)}")
            return []
    return []


def save_tracker_to_csv():
    """Save tracked stocks to CSV file"""
    try:
        if st.session_state.tracked_stocks:
            df = pd.DataFrame(st.session_state.tracked_stocks)
            df.to_csv(TRACKER_CSV_FILE, index=False)
            logger.info(f"Saved {len(st.session_state.tracked_stocks)} stocks to {TRACKER_CSV_FILE}")
        else:
            # Create empty CSV with headers if no stocks (for cloud persistence)
            empty_df = pd.DataFrame(columns=['symbol', 'name', 'purchase_price', 'shares', 'purchase_date', 'target_price', 'allocation_aed'])
            empty_df.to_csv(TRACKER_CSV_FILE, index=False)
            logger.info(f"Created empty tracker CSV file")
    except Exception as e:
        logger.error(f"Error saving tracker CSV: {str(e)}")
        # Try to create the file if it doesn't exist (for cloud environments)
        try:
            empty_df = pd.DataFrame(columns=['symbol', 'name', 'purchase_price', 'shares', 'purchase_date', 'target_price', 'allocation_aed'])
            empty_df.to_csv(TRACKER_CSV_FILE, index=False)
        except Exception as e2:
            logger.error(f"Failed to create CSV file: {str(e2)}")


def initialize_tracker():
    """Initialize the stock tracker in session state and load from CSV"""
    if 'tracked_stocks' not in st.session_state:
        st.session_state.tracked_stocks = load_tracker_from_csv()
        logger.info(f"Initialized tracker with {len(st.session_state.tracked_stocks)} stocks from CSV")


def add_to_tracker(stock_data, purchase_price, shares, purchase_date):
    """Add a stock to the tracker and save to CSV"""
    tracked_stock = {
        'symbol': stock_data['symbol'],
        'name': stock_data.get('name', stock_data['symbol']),
        'purchase_price': purchase_price,
        'shares': shares,
        'purchase_date': purchase_date,
        'target_price': purchase_price * (1 + TARGET_PERCENTAGE / 100),
        'allocation_aed': shares * purchase_price * USD_TO_AED
    }
    st.session_state.tracked_stocks.append(tracked_stock)
    save_tracker_to_csv()  # Save to CSV immediately
    logger.info(f"Added {stock_data['symbol']} to tracker: {shares} shares @ ${purchase_price:.2f}")


def get_current_price(symbol):
    """Get current price for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return None
    except Exception as e:
        logger.error(f"Error fetching current price for {symbol}: {str(e)}")
        return None


def calculate_performance(tracked_stock):
    """Calculate performance metrics for a tracked stock"""
    current_price = get_current_price(tracked_stock['symbol'])
    if current_price is None:
        return None
    
    purchase_price = tracked_stock['purchase_price']
    target_price = tracked_stock['target_price']
    
    # Calculate percentage change from purchase
    pct_change = ((current_price - purchase_price) / purchase_price) * 100
    
    # Calculate percentage to target
    pct_to_target = ((target_price - current_price) / current_price) * 100 if current_price > 0 else 0
    
    # Calculate profit/loss
    profit_loss = (current_price - purchase_price) * tracked_stock['shares']
    profit_loss_aed = profit_loss * USD_TO_AED
    
    # Current value
    current_value = current_price * tracked_stock['shares']
    current_value_aed = current_value * USD_TO_AED
    
    return {
        'current_price': current_price,
        'pct_change': pct_change,
        'pct_to_target': pct_to_target,
        'profit_loss': profit_loss,
        'profit_loss_aed': profit_loss_aed,
        'current_value': current_value,
        'current_value_aed': current_value_aed,
        'target_price': target_price,
        'target_reached': current_price >= target_price
    }


def display_tracker_page():
    """Display the Stock Tracker page"""
    st.header("📊 Stock Tracker")
    st.markdown("**Track your purchased stocks and monitor performance**")
    
    # Show info about CSV persistence
    if os.path.exists(TRACKER_CSV_FILE):
        st.info(f"💾 Tracker data is saved to `{TRACKER_CSV_FILE}` and will persist across app restarts.")
    
    initialize_tracker()
    
    if not st.session_state.tracked_stocks:
        st.info("📝 No stocks in tracker yet. Buy stocks from the Scanner tab and add them here!")
        return
    
    # Summary metrics
    total_invested_aed = sum(stock['allocation_aed'] for stock in st.session_state.tracked_stocks)
    
    # Calculate total performance
    total_profit_loss_aed = 0
    total_current_value_aed = 0
    
    tracker_data = []
    for stock in st.session_state.tracked_stocks:
        perf = calculate_performance(stock)
        if perf:
            tracker_data.append({
                **stock,
                **perf
            })
            total_profit_loss_aed += perf['profit_loss_aed']
            total_current_value_aed += perf['current_value_aed']
    
    if not tracker_data:
        st.warning("Unable to fetch current prices. Please try again later.")
        return
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Invested (AED)", format_currency(total_invested_aed, "AED"))
    
    with col2:
        st.metric("Current Value (AED)", format_currency(total_current_value_aed, "AED"))
    
    with col3:
        profit_color = "normal" if total_profit_loss_aed >= 0 else "inverse"
        st.metric("Total P&L (AED)", format_currency(total_profit_loss_aed, "AED"), delta=f"{((total_current_value_aed - total_invested_aed) / total_invested_aed * 100):.2f}%")
    
    with col4:
        targets_reached = sum(1 for data in tracker_data if data.get('target_reached', False))
        st.metric("Targets Reached", f"{targets_reached}/{len(tracker_data)}")
    
    st.divider()
    
    # Display tracker table
    df_tracker = pd.DataFrame(tracker_data)
    
    # Prepare display columns
    display_data = df_tracker[[
        'symbol', 'name', 'purchase_date', 'purchase_price', 'shares',
        'current_price', 'pct_change', 'target_price', 'pct_to_target',
        'profit_loss_aed', 'current_value_aed', 'target_reached'
    ]].copy()
    
    display_data.columns = [
        'Symbol', 'Company Name', 'Purchase Date', 'Purchase Price (USD)', 'Shares',
        'Current Price (USD)', 'Change %', 'Target Price (USD)', 'To Target %',
        'P&L (AED)', 'Current Value (AED)', 'Target Reached'
    ]
    
    # Format columns
    display_data['Purchase Price (USD)'] = display_data['Purchase Price (USD)'].apply(lambda x: format_currency(x, "USD"))
    display_data['Current Price (USD)'] = display_data['Current Price (USD)'].apply(lambda x: format_currency(x, "USD"))
    display_data['Target Price (USD)'] = display_data['Target Price (USD)'].apply(lambda x: format_currency(x, "USD"))
    display_data['Change %'] = display_data['Change %'].apply(lambda x: f"{x:+.2f}%")
    display_data['To Target %'] = display_data['To Target %'].apply(lambda x: f"{x:.2f}%")
    display_data['P&L (AED)'] = display_data['P&L (AED)'].apply(lambda x: format_currency(x, "AED"))
    display_data['Current Value (AED)'] = display_data['Current Value (AED)'].apply(lambda x: format_currency(x, "AED"))
    display_data['Target Reached'] = display_data['Target Reached'].map({True: '✅ Yes', False: '⏳ No'})
    
    # Style the dataframe
    def style_tracker(row):
        """Style tracker rows based on performance"""
        styles = [''] * len(row)
        change_pct = float(row['Change %'].replace('%', '').replace('+', ''))
        
        if change_pct >= TARGET_PERCENTAGE:
            # Target reached - green
            return ['background-color: #d4edda'] * len(row)
        elif change_pct >= 0:
            # Positive but not at target - light green
            return ['background-color: #d1ecf1'] * len(row)
        else:
            # Negative - light red
            return ['background-color: #f8d7da'] * len(row)
    
    styled_df = display_data.style.apply(style_tracker, axis=1)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Remove stock option
    st.divider()
    st.subheader("Remove Stock from Tracker")
    
    if st.session_state.tracked_stocks:
        symbols = [f"{s['symbol']} - {s['name']}" for s in st.session_state.tracked_stocks]
        selected_to_remove = st.selectbox(
            "Select stock to remove:",
            options=symbols,
            key="remove_stock_select"
        )
        
        if st.button("🗑️ Remove from Tracker", type="secondary"):
            # Extract symbol from selection
            symbol_to_remove = selected_to_remove.split(' - ')[0]
            st.session_state.tracked_stocks = [
                s for s in st.session_state.tracked_stocks 
                if s['symbol'] != symbol_to_remove
            ]
            save_tracker_to_csv()  # Save to CSV immediately
            logger.info(f"Removed {symbol_to_remove} from tracker")
            st.success(f"Removed {symbol_to_remove} from tracker!")
            st.rerun()


def display_scanner_sidebar():
    """Display the sidebar for scanner page"""
    with st.sidebar:
        st.header("Investment Parameters")
        st.metric("Total Capital", format_currency(TOTAL_CAPITAL_AED, "AED"))
        st.metric("Allocation per Stock", format_currency(ALLOCATION_PER_STOCK_AED, "AED"))
        st.metric("Exchange Rate", f"1 USD = {USD_TO_AED} AED")
        st.metric("Target Profit %", f"{TARGET_PERCENTAGE}%")
        logger.info("Sidebar parameters displayed")
        
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            logger.info("Refresh button clicked - clearing cache")
            st.cache_data.clear()
            st.rerun()


def main():
    logger.info("=" * 50)
    logger.info("Starting DJIA Stock Investment Scanner App")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize tracker
        initialize_tracker()
        
        # Header
        st.title("📈 DJIA Stock Investment Scanner")
        st.markdown("**Automated stock scanning based on 52-week high drops and daily price movements**")
        logger.info("Header displayed")
        
        # Create tabs
        tab1, tab2 = st.tabs(["🔍 Stock Scanner", "📊 Stock Tracker"])
        
        with tab1:
            display_scanner_sidebar()
            
            # Fetch and analyze stocks
            logger.info("Calling analyze_stocks()...")
            df, data_list = analyze_stocks()
            
            if df is None:
                logger.error("analyze_stocks() returned None - exiting")
                return
            
            logger.info(f"Successfully received DataFrame with {len(df)} stocks")
            
            # Determine buy stocks
            logger.info("Calling determine_buy_stocks()...")
            buy_stocks = determine_buy_stocks(df)
            logger.info(f"Buy stocks determined: {len(buy_stocks)} stock(s)")
            
            # Summary Metrics
            logger.info("Displaying summary metrics...")
            col1, col2, col3, col4 = st.columns(4)
        
            with col1:
                st.metric("Total Capital (AED)", format_currency(TOTAL_CAPITAL_AED, "AED"))
            
            with col2:
                # Calculate total allocated: today's recommendations + already tracked stocks
                total_allocated_today = len(buy_stocks) * ALLOCATION_PER_STOCK_AED if not buy_stocks.empty else 0
                total_allocated_tracked = sum(stock['allocation_aed'] for stock in st.session_state.tracked_stocks)
                total_allocated = total_allocated_today + total_allocated_tracked
                st.metric("Allocated Today", format_currency(total_allocated_today, "AED"))
            
            with col3:
                # Available capital = Total - Already invested in tracker
                available = TOTAL_CAPITAL_AED - total_allocated_tracked
                st.metric("Available Capital", format_currency(available, "AED"))
                if total_allocated_tracked > 0:
                    st.caption(f"({total_allocated_tracked:,.0f} AED already invested)")
            
            with col4:
                st.metric("Stocks to Buy", len(buy_stocks) if not buy_stocks.empty else 0)
            
            st.divider()
            
            # Buy Recommendations Section (with add to tracker functionality)
            logger.info("Displaying buy recommendations section...")
            if not buy_stocks.empty:
                logger.info(f"Displaying {len(buy_stocks)} buy recommendation(s)")
                st.header("🟢 Buy Recommendations")
                st.success(f"**{len(buy_stocks)} stock(s) identified for purchase today**")
                
                # Display buy recommendations in a styled table
                buy_display = buy_stocks[[
                    'symbol', 'name', 'current_price', 'prev_close', 
                    'price_change_pct', 'high_52w', 'drop_from_52w',
                    'shares_to_buy', 'total_cost_usd', 'total_cost_aed'
                ]].copy()
                
                buy_display.columns = [
                    'Symbol', 'Company Name', 'Current Price (USD)', 'Prev Close (USD)',
                    'Price Change %', '52W High (USD)', 'Drop from 52W %',
                    'Shares to Buy', 'Total Cost (USD)', 'Total Cost (AED)'
                ]
                
                # Format the display
                buy_display['Current Price (USD)'] = buy_display['Current Price (USD)'].apply(lambda x: format_currency(x, "USD"))
                buy_display['Prev Close (USD)'] = buy_display['Prev Close (USD)'].apply(lambda x: format_currency(x, "USD"))
                buy_display['52W High (USD)'] = buy_display['52W High (USD)'].apply(lambda x: format_currency(x, "USD"))
                buy_display['Price Change %'] = buy_display['Price Change %'].apply(lambda x: f"{x:.2f}%")
                buy_display['Drop from 52W %'] = buy_display['Drop from 52W %'].apply(lambda x: f"{x:.2f}%")
                buy_display['Total Cost (USD)'] = buy_display['Total Cost (USD)'].apply(lambda x: format_currency(x, "USD"))
                buy_display['Total Cost (AED)'] = buy_display['Total Cost (AED)'].apply(lambda x: format_currency(x, "AED"))
                
                st.dataframe(
                    buy_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Total investment summary
                total_investment_usd = buy_stocks['total_cost_usd'].sum()
                total_investment_aed = buy_stocks['total_cost_aed'].sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Investment (USD)", format_currency(total_investment_usd, "USD"))
                with col2:
                    st.metric("Total Investment (AED)", format_currency(total_investment_aed, "AED"))
                
                st.divider()
                
                # Add to Tracker section
                st.subheader("📌 Add to Stock Tracker")
                st.markdown("Select stocks to add to your tracker:")
                
                for idx, stock in buy_stocks.iterrows():
                    with st.expander(f"➕ Add {stock['symbol']} - {stock['name']} to Tracker"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Purchase Price:** {format_currency(stock['current_price'], 'USD')}")
                            st.write(f"**Shares:** {int(stock['shares_to_buy'])}")
                            st.write(f"**Total Cost:** {format_currency(stock['total_cost_aed'], 'AED')}")
                        with col2:
                            target_price = stock['current_price'] * (1 + TARGET_PERCENTAGE / 100)
                            st.write(f"**Target Price:** {format_currency(target_price, 'USD')}")
                            st.write(f"**Target Profit:** {TARGET_PERCENTAGE}%")
                            purchase_date = st.date_input(
                                "Purchase Date:",
                                value=datetime.now().date(),
                                key=f"purchase_date_{stock['symbol']}"
                            )
                        
                        if st.button(f"Add {stock['symbol']} to Tracker", key=f"add_{stock['symbol']}"):
                            # Get stock info from original df
                            stock_info = df[df['symbol'] == stock['symbol']].iloc[0].to_dict()
                            add_to_tracker(
                                stock_info,
                                stock['current_price'],
                                int(stock['shares_to_buy']),
                                purchase_date.strftime('%Y-%m-%d')
                            )
                            st.success(f"✅ {stock['symbol']} added to tracker!")
                            st.rerun()
                
                st.divider()
            else:
                logger.info("No buy stocks found - displaying info message")
                st.info("ℹ️ No stocks meet the buy criteria today. Check back later!")
                st.divider()
            
            # All DJIA Stocks Table
            logger.info("Displaying all DJIA stocks table...")
            st.header("📊 All DJIA Stocks Analysis")
            
            # Prepare display DataFrame
            display_df = df[[
                'symbol', 'name', 'current_price', 'prev_close',
                'price_change', 'price_change_pct', 'high_52w', 'drop_from_52w',
                'meets_condition_a', 'meets_condition_b', 'meets_both_conditions'
            ]].copy()
            
            display_df.columns = [
                'Symbol', 'Company Name', 'Current Price (USD)', 'Prev Close (USD)',
                'Price Change (USD)', 'Price Change %', '52W High (USD)', 'Drop from 52W %',
                'Condition A (Dropped from 52W)', 'Condition B (Rising Today)', 'Buy Signal'
            ]
            
            # Format currency columns
            display_df['Current Price (USD)'] = display_df['Current Price (USD)'].apply(lambda x: format_currency(x, "USD"))
            display_df['Prev Close (USD)'] = display_df['Prev Close (USD)'].apply(lambda x: format_currency(x, "USD"))
            display_df['Price Change (USD)'] = display_df['Price Change (USD)'].apply(lambda x: format_currency(x, "USD"))
            display_df['52W High (USD)'] = display_df['52W High (USD)'].apply(lambda x: format_currency(x, "USD"))
            display_df['Price Change %'] = display_df['Price Change %'].apply(lambda x: f"{x:.2f}%")
            display_df['Drop from 52W %'] = display_df['Drop from 52W %'].apply(lambda x: f"{x:.2f}%")
            
            # Convert boolean columns to Yes/No
            display_df['Condition A (Dropped from 52W)'] = display_df['Condition A (Dropped from 52W)'].map({True: '✅ Yes', False: '❌ No'})
            display_df['Condition B (Rising Today)'] = display_df['Condition B (Rising Today)'].map({True: '✅ Yes', False: '❌ No'})
            display_df['Buy Signal'] = display_df['Buy Signal'].map({True: '🟢 BUY', False: '⚪ No'})
            
            # Style the dataframe with colors
            def highlight_buy(row):
                """Highlight buy signals in green"""
                if row['Buy Signal'] == '🟢 BUY':
                    return ['background-color: #d4edda'] * len(row)
                return [''] * len(row)
            
            styled_df = display_df.style.apply(highlight_buy, axis=1)
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Footer
            st.divider()
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: Yahoo Finance")
        
        with tab2:
            display_tracker_page()
        
        logger.info("=" * 50)
        logger.info("App rendering completed successfully")
        logger.info("=" * 50)
    
    except Exception as e:
        logger.error(f"Fatal error in main(): {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()

