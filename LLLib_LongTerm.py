import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

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

                # get date range
                start_date = "1980-01-01"
                end_date = dt.datetime.today()

                # get stock data for each symbol

                for symbol in symbolList:

                    stock_Data_forSymbol = yf.download(symbol, start=start_date, end=end_date)

                    print(stock_Data_forSymbol)







