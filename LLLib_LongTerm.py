import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
from dateutil.relativedelta import relativedelta
import traceback


import LLLib_Charles_Schwab as cswab




class LLLib_LongTerm:

    class Get_Stuff:

        class From_LootLoaderDatabase:
        
            def get_Symbol_List():

                #pull symbol list from lootloaderdatabase

                symbolList = cswab.Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()

                return symbolList

        class From_YFinance:            


            def get_Historical_Day_Data_for_All_Symbols():

                
                # get symbol list

                symbolList = LLLib_LongTerm.Get_Stuff.From_LootLoaderDatabase.get_Symbol_List()

                # get date range over 25 years
                todaysDate = dt.datetime.today()
                date_minus_25_years = todaysDate - relativedelta(years=25)
                start_date = date_minus_25_years
                end_date = dt.datetime.today()

                # get stock data for each symbol

                for symbol in symbolList:
                    
                    try:
                        stock_Data_forSymbol_df = yf.download(symbol, start=start_date, end=end_date)

                        #print(stock_Data_forSymbol_df)
                        return stock_Data_forSymbol_df
                    except:
                        print("Error occurred trying to get data for symbol "+symbol)
                        print("symbol data: "+stock_Data_forSymbol_df)
                        print(traceback.format_exc())

        class Files:

            def load_tickers(file_path):
                if not os.path.exists(file_path):
                    print(f"[WARN] Ticker file not found: {file_path}")
                    # Create an empty template if you like:
                    pd.DataFrame({"Ticker": []}).to_csv(file_path, index=False)
                    return []
                df = pd.read_csv(file_path)
                if 'Ticker' not in df.columns:
                    raise ValueError("Ticker file must contain a column named 'Ticker'")
                return df['Ticker'].dropna().astype(str).tolist()





