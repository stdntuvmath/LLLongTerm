import pandas as pd

# (A) Load index component lists from public sources
sp500 = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")  # from DataHub :contentReference[oaicite:4]{index=4}
nasdaq100 = pd.read_html("https://www.slickcharts.com/nasdaq100")[0]  # from SlickCharts :contentReference[oaicite:5]{index=5}
dow30 = pd.read_html("https://www.stockanalysis.com/list/dow-jones-stocks/")[0]  # from StockAnalysis :contentReference[oaicite:6]{index=6}
nyse = pd.read_csv("https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv")  # from DataHub :contentReference[oaicite:7]{index=7}

# (B) Extract ticker columns, rename for consistency
sp500_tickers = sp500[['Symbol']].rename(columns={'Symbol':'Ticker'})
nasdaq100_tickers = nasdaq100[['Symbol']].rename(columns={'Symbol':'Ticker'})
dow30_tickers = dow30[['Symbol']].rename(columns={'Symbol':'Ticker'})
nyse_tickers = nyse[['ACT Symbol']].rename(columns={'ACT Symbol':'Ticker'})

# (C) Combine and drop duplicates
combined = pd.concat([sp500_tickers, nasdaq100_tickers, dow30_tickers, nyse_tickers], ignore_index=True)
combined = combined.drop_duplicates().sort_values(by='Ticker').reset_index(drop=True)

# (D) Save to your local folder
output_path = r"C:\Users\stdnt\Desktop\LootLoader\LongTerm\stock_list_all.csv"
combined.to_csv(output_path, index=False)
print(f"Saved combined ticker list to {output_path}. Total tickers: {len(combined)}")
