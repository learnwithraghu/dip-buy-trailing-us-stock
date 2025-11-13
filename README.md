# DJIA Stock Investment Scanner App

A modern Streamlit application that automatically scans all 30 Dow Jones Industrial Average (DJIA) stocks to identify buy opportunities based on specific investment criteria.

## Features

- **Real-time Stock Data**: Fetches live stock prices and 52-week highs for all DJIA stocks
- **Automated Scanning**: Identifies stocks that have fallen from 52-week highs and are rising from the previous day's close
- **Smart Buy Logic**:
  - Buys up to 3 stocks if the top-ranked stock has fallen more than -10% from 52-week high
  - Buys only 1 stock if the top-ranked stock has fallen less than or equal to -10% from 52-week high
- **Currency Handling**:
  - All prices displayed in USD
  - Buy recommendations calculated based on AED capital allocation (AED 1,250 per stock)
- **Modern UI**: Clean, color-coded interface with clear buy signals

## Investment Strategy

### Capital Allocation

- **Total Capital**: AED 50,000
- **Allocation per Stock**: AED 1,250 (40 equal parts)

### Buy Conditions

Stocks must meet both conditions:

1. **Condition A**: Stock has fallen from its 52-week high closing price
2. **Condition B**: Stock is trading above its previous day's closing price

### Buy Decision Algorithm

- If the top-ranked stock (by % drop from 52W high) has fallen **more than -10%** from 52-week high:
  - Buy **maximum 3 stocks** (all must meet both conditions)
- If the top-ranked stock has fallen **less than or equal to -10%** from 52-week high:
  - Buy **only 1 stock** (must meet both conditions)

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or navigate to the project directory**:

   ```bash
   cd streamlit_example_usa
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:

   **On macOS/Linux**:

   ```bash
   source venv/bin/activate
   ```

   **On Windows**:

   ```bash
   venv\Scripts\activate
   ```

   You should see `(venv)` in your terminal prompt when activated.

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:

   ```bash
   streamlit run app.py
   ```

6. **Access the app**:
   The app will automatically open in your default web browser at `http://localhost:8501`

**Note**: Remember to activate your virtual environment each time you work on the project. To deactivate the virtual environment, simply run `deactivate`.

## Usage

1. **View Summary Metrics**: The top section shows total capital, allocated amount, available capital, and number of stocks to buy

2. **Check Buy Recommendations**: The green "Buy Recommendations" section displays:
   - Stock symbols and company names
   - Current prices (in USD)
   - Price changes and 52-week high information
   - Number of shares to buy
   - Total cost in both USD and AED

3. **Review All Stocks**: The "All DJIA Stocks Analysis" table shows:
   - All 30 DJIA stocks with their current status
   - Color-coded rows (green for buy signals)
   - Whether each stock meets the buy conditions

4. **Refresh Data**: Click the "🔄 Refresh Data" button in the sidebar to fetch the latest stock prices

## Technical Details

- **Data Source**: Yahoo Finance (via `yfinance` library)
- **Data Refresh**: Manual refresh button + automatic on app load
- **Caching**: Data is cached for 5 minutes to improve performance
- **Exchange Rate**: USD to AED conversion rate (default: 3.67, can be updated in code)

## Project Structure

```text
streamlit_example_usa/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── venv/              # Virtual environment (created during setup, not in git)
```

## Dependencies

- `streamlit>=1.28.0` - Web application framework
- `yfinance>=0.2.28` - Yahoo Finance data fetching
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computations

## Notes

- Stock data is fetched from Yahoo Finance, which may have rate limits
- The app includes error handling for API failures
- All prices are displayed in USD, but buy calculations use AED amounts
- The exchange rate (USD to AED) can be updated in the `USD_TO_AED` constant in `app.py`

## Deployment

### Deploy to Streamlit Cloud

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: DJIA Stock Scanner App"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set the main file path to: `app.py`
   - Click "Deploy!"

3. **Important Notes for Cloud Deployment**:
   - The `stock_tracker.csv` file will be created automatically in the cloud environment
   - Your tracker data will persist as long as the app is deployed
   - The CSV file is stored in the app's working directory
   - Make sure `.gitignore` includes `stock_tracker.csv` so it's not committed to GitHub

4. **After Deployment**:
   - Your app will be available at: `https://your-app-name.streamlit.app`
   - The tracker data will persist across sessions
   - You can share the app URL with others (they'll have separate tracker data)

## License

This project is for personal investment analysis purposes.
