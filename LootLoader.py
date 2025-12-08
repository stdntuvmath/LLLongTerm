



# ==========================
# === Dependencies ===
# ==========================


import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import json
import Get_Current_List_of_Stocks
import LLLib_LongTerm as lib



# ==========================
# === Get List Of Stocks ===
# ==========================

# creates stock_list_all.csv
Get_Current_List_of_Stocks.Run()

# Parse stock_list_all.csv to symbol list
stockData = lib.LLLib_LongTerm.Get_Stuff.From_YFinance.get_Historical_Day_Data_for_All_Symbols()







