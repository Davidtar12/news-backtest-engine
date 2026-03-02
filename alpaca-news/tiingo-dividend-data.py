import requests
import datetime
from dotenv import load_dotenv
import os
load_dotenv()


# ==========================
# CONFIG
# ==========================
TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_API_KEY}"
}

tickers = ["TIGO", "MO", "PM", "T", "CPA", "BAP", "CVS", "AU"]

# ==========================
# TEST API CONNECTIVITY
# ==========================
def test_api_connection():
    """Test if the API key works"""
    url = f"https://api.tiingo.com/api/test?token={TIINGO_API_KEY}"
    r = requests.get(url)
    print(f"API Test Status: {r.status_code}")
    if r.status_code == 200:
        print(f"API Response: {r.json()}")
        return True
    else:
        print(f"API Error: {r.text}")
        return False

# ==========================
# GET DIVIDENDS - CORRECT CORPORATE ACTIONS ENDPOINT
# ==========================
def get_dividends(symbol):
    """
    Use the correct Tiingo corporate actions distributions endpoint
    """
    url = f"https://api.tiingo.com/tiingo/corporate-actions/{symbol}/distributions"
    
    # Try both authentication methods
    auth_methods = [
        {"url": url, "headers": headers},  # Header auth
        {"url": f"{url}?token={TIINGO_API_KEY}", "headers": {"Content-Type": "application/json"}}  # URL auth
    ]
    
    for i, auth in enumerate(auth_methods, 1):
        print(f"  Trying auth method {i} for {symbol}...")
        r = requests.get(auth["url"], headers=auth["headers"])
        
        if r.status_code == 200:
            try:
                data = r.json()
                if data:
                    print(f"    ✅ Success! Found {len(data)} distribution records")
                    return data
                else:
                    print(f"    ⚠️ Empty response")
            except Exception as e:
                print(f"    ❌ JSON parsing error: {e}")
        else:
            print(f"    ❌ Auth method {i}: {r.status_code} - {r.text[:150]}")
    
    return []

# ==========================
# GET DISTRIBUTION YIELD (BONUS)
# ==========================
def get_distribution_yield(symbol):
    """
    Get historical distribution yield data (optional)
    """
    url = f"https://api.tiingo.com/tiingo/corporate-actions/{symbol}/distribution-yield"
    
    # Try header auth first
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        try:
            data = r.json()
            if data:
                latest = data[-1] if isinstance(data, list) else data
                return latest.get("yield", 0)
        except:
            pass
    
    return None

# ==========================
# TEST TICKER EXISTS
# ==========================
def test_ticker_exists(symbol):
    """Test if ticker exists using basic ticker info"""
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}"
    
    # Try both auth methods
    auth_methods = [
        {"url": url, "headers": headers},
        {"url": f"{url}?token={TIINGO_API_KEY}", "headers": {"Content-Type": "application/json"}}
    ]
    
    for auth in auth_methods:
        r = requests.get(auth["url"], headers=auth["headers"])
        if r.status_code == 200:
            try:
                data = r.json()
                print(f"  ✅ {symbol} exists: {data.get('name', 'Unknown')}")
                return True
            except:
                pass
    
    print(f"  ❌ {symbol} not found or not accessible")
    return False

# ==========================
# PRICE CHANGE CALC
# ==========================
def get_price_change(symbol, ex_date):
    """
    Download prices from Tiingo daily endpoint.
    Compare close at ex-date vs close 3 trading days later.
    """
    try:
        # Calculate end date
        ex_date_obj = datetime.datetime.strptime(ex_date, "%Y-%m-%d").date()
        end_date = ex_date_obj + datetime.timedelta(days=10)
        
        # Build URL with parameters
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = f"startDate={ex_date}&endDate={end_date}"
        
        # Try both authentication methods
        auth_methods = [
            {"url": f"{url}?{params}", "headers": headers},
            {"url": f"{url}?{params}&token={TIINGO_API_KEY}", "headers": {"Content-Type": "application/json"}}
        ]
        
        for auth in auth_methods:
            r = requests.get(auth["url"], headers=auth["headers"])
            
            if r.status_code == 200:
                prices = r.json()
                if len(prices) < 4:
                    print(f"    ⚠️ Not enough price data (got {len(prices)} days)")
                    return None
                
                ex_close = prices[0]["close"]
                after_close = prices[3]["close"]  # 3 trading days later
                change_pct = (after_close - ex_close) / ex_close * 100
                
                return ex_close, after_close, round(change_pct, 2)
        
    except Exception as e:
        print(f"    ❌ Price calculation error: {e}")
    
    return None

# ==========================
# MAIN
# ==========================
def main():
    print("🔧 Testing API connection...")
    if not test_api_connection():
        print("❌ API connection failed. Please check your API key.")
        return
    
    print("\n" + "="*70)
    print("DIVIDEND ANALYSIS - TIINGO CORPORATE ACTIONS API")
    print("="*70)
    
    for ticker in tickers:
        print(f"\n🔍 Processing {ticker}...")
        
        # Test if ticker exists
        if not test_ticker_exists(ticker):
            continue
        
        # Get distribution (dividend) data
        print(f"  📊 Fetching distributions for {ticker}...")
        distributions = get_dividends(ticker)
        
        if not distributions:
            print(f"  ❌ No distribution data found for {ticker}")
            continue
        
        # Get distribution yield (optional)
        dist_yield = get_distribution_yield(ticker)
        if dist_yield:
            print(f"  📈 Current distribution yield: {dist_yield:.2f}%")
        
        # Process the last 2 distributions
        try:
            # Sort by date (most recent first) - handle different date field names
            def get_date(dist):
                return (dist.get("exDate") or 
                       dist.get("ex_date") or 
                       dist.get("date") or 
                       dist.get("effectiveDate") or "")
            
            distributions_sorted = sorted(distributions, 
                                        key=get_date, 
                                        reverse=True)[:2]
            
            print(f"  ✅ Found {len(distributions)} total distributions, analyzing last 2:")
            
            for i, dist in enumerate(distributions_sorted, 1):
                # Handle different possible field names from corporate actions API
                ex_date = (dist.get("exDate") or 
                          dist.get("ex_date") or 
                          dist.get("date") or 
                          dist.get("effectiveDate"))
                
                cash_amount = (dist.get("cashAmount") or 
                             dist.get("amount") or 
                             dist.get("dividend") or 
                             dist.get("cash", 0.0))
                
                payment_date = (dist.get("payDate") or 
                              dist.get("paymentDate") or 
                              dist.get("pay_date") or "N/A")
                
                dist_type = dist.get("type", "dividend")
                
                print(f"\n    📌 {ticker} Distribution #{i}")
                print(f"       Type: {dist_type}")
                print(f"       Ex-Date: {ex_date}")
                print(f"       Payment: {payment_date}")
                print(f"       Amount: ${cash_amount:.4f}")
                
                # Show raw data for debugging (first distribution only)
                if i == 1:
                    print(f"       [Debug] Available fields: {list(dist.keys())}")
                
                # Calculate price impact if we have ex-date
                if ex_date and cash_amount > 0:
                    print(f"    💰 Calculating price impact...")
                    price_change = get_price_change(ticker, ex_date)
                    if price_change:
                        ex_close, after_close, pct = price_change
                        print(f"       Price @Ex-Date: ${ex_close:.2f}")
                        print(f"       Price @+3D: ${after_close:.2f}")
                        print(f"       Change: {pct:+.2f}%")
                        
                        # Calculate dividend yield for this specific distribution
                        div_yield = (cash_amount / ex_close) * 100
                        print(f"       Distribution Yield: {div_yield:.2f}%")
                        
                        # Compare to theoretical drop
                        theoretical_drop = (cash_amount / ex_close) * 100
                        actual_vs_theoretical = pct + theoretical_drop
                        print(f"       Expected drop: -{theoretical_drop:.2f}%")
                        print(f"       Excess return: {actual_vs_theoretical:+.2f}%")
                
        except Exception as e:
            print(f"  ❌ Error processing distributions for {ticker}: {e}")
            # Show first few distributions for debugging
            if distributions:
                print(f"  🔍 Sample distribution data: {distributions[0]}")

if __name__ == "__main__":
    main()
