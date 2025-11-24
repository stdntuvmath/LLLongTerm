#*************************************************
# NAME: LootLoader Library for Charles Schwab API.
#       This code embodies a wrapper for the
#       unofficial wrapper around the original API.
# AUTHOR: Brandon Turner
#
# PURPOSE: This library is for LootLoader to work
#           with Charles Schwab API. 
#
#
#*************************************************



import os
import ast
import webbrowser
import datetime
import json
import base64
import mysql.connector
import requests as req
import numpy as np
import traceback
import regex as re
import pandas as pd
import time as tme
from dateutil.tz import tz
from datetime import date
from datetime import datetime as dt
from datetime import timedelta as td
import datetime
import colorama as corama
import schwab
import mysql
import httpx
import math
import decimal
import subprocess
import flask

from schwab.orders.common import Duration, Session
from schwab.orders.common import OrderType, StopType
from schwab.orders.generic import OrderBuilder
from schwab.orders.equities import equity_buy_market

global windowsUsername
windowsUsername = os.getenv('USERNAME')#get the windows username of the user

class Account_Management:

    global windowsUsername
    windowsUsername = os.getenv('USERNAME')#get the windows username of the user
    
    def Get_Account_Number():

        try:

            accountHash = Account_Management.Get_Account_Hash()
            #print(accountHash)

            auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
            account = auth.get_account(accountHash)

            file = account.json()
            # print(account)
            #print(json.dumps(account.json(), indent=4))
                

            for headings in file:
                if(str(headings) == 'securitiesAccount'):
                    return file[headings]['accountNumber']

        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)

    def Get_Account_Balance_Available_For_Trading():

        try:

            accountHash = Account_Management.Get_Account_Hash()
            #print(accountHash)

            auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
            account = auth.get_account(accountHash)

            file = account.json()
            # print(account)
            #print(json.dumps(account.json(), indent=4))
                

            for headings in file:
                if(str(headings) == 'securitiesAccount'):
                    return file[headings]['currentBalances']['cashAvailableForTrading']

        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)        
    
    def Get_Account_Total_Amount():

        try:

            accountHash = Account_Management.Get_Account_Hash()
            #print(accountHash)

            auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
            account = auth.get_account(accountHash)

            file = account.json()
            # print(account)
            #print(json.dumps(account.json(), indent=4))
                

            for headings in file:
                if(str(headings) == 'securitiesAccount'):
                    return file[headings]['initialBalances']['cashBalance']

        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)        
    
    def Get_Account_Balance_Available_For_Withdrawal():

        try:

            accountHash = Account_Management.Get_Account_Hash()
            #print(accountHash)

            auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
            account = auth.get_account(accountHash)

            file = account.json()
            # print(account)
            #print(json.dumps(account.json(), indent=4))
                

            for headings in file:
                if(str(headings) == 'securitiesAccount'):
                    return file[headings]['initialBalances']['cashAvailableForWithdrawal']

        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)        

    def Get_Account_Hash():

        try:
            appkey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_appKey()
            skey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_secretKey()
            cb_url = "https://127.0.0.1"
            tokenPath = "tokens.json"

            auth = schwab.auth.easy_client(appkey, skey, cb_url, tokenPath)

            response = auth.get_account_numbers()

            assert response.status_code == httpx.codes.OK

            account_hash = response.json()[0]['hashValue']

            return account_hash

        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)


class Authentication_Management:

    class Authentication_MyVersion:

            class Get_Keys_From_CredentialsFile:     

                def Get_appKey():
                    try:

                        #turn credentials.json file content into string by loading file into dumps function
                        f = open('credentials.json')
                        jsonString = json.dumps(json.load(f))
                        f.close()
                        #make dictionary file from string and iterate through dictionary to get the appKey
                        jsonDictionary = json.loads(jsonString)

                        for credential in jsonDictionary['creds']:
                            appKey = credential.get('appKey')

                        return appKey
                    
                    except:
                        error = traceback.format_exc()
                        Logging.error(error)
                
                def Get_secretKey():
                    try:

                        #turn credentials.json file content into string by loading file into dumps function
                        f = open('credentials.json')
                        jsonString = json.dumps(json.load(f))
                        f.close()
                        #make dictionary file from string and iterate through dictionary to get the secretKey
                        jsonDictionary = json.loads(jsonString)

                        for credential in jsonDictionary['creds']:
                            secretKey = credential.get('secretKey')

                        return secretKey
                    
                    except:
                        error = traceback.format_exc()
                        Logging.error(error)


                def Get_LootLoader_Password():
                    try:

                        #turn credentials.json file content into string by loading file into dumps function
                        f = open('credentials.json')
                        jsonString = json.dumps(json.load(f))
                        f.close()
                        #make dictionary file from string and iterate through dictionary to get the appKey
                        jsonDictionary = json.loads(jsonString)

                        for credential in jsonDictionary['passwords']:
                            password = credential.get('lootloader_password')

                        return password
                    
                    except:
                        error = traceback.format_exc()
                        Logging.error(error)



            class Get_Authentication_Tokens:

                def Get_accessToken():

                    try:
                        #turn credentials.json file content into string by loading file into dumps function
                        with open('tokens.dict') as f:
                            data = f.read()
                        f.close()
                        # data = data.replace("'", "\"")
                        #data = data.replace("{", "'{")
                        #data = data.replace("}", "}'")

                        data = ast.literal_eval(data)#change string to a dictionary
                        #print(type(data))
                        #print(str(data))


                        for stuff in data:
                            if(stuff == "access_token"):
                                return data[stuff]
                    
                    except:
                        error = traceback.format_exc()
                        Logging.error(error)


                def Get_accessToken_startTime():

                    try:
                        #turn credentials.json file content into string by loading file into dumps function
                        with open('tokens.dict') as f:
                            data = f.read()
                        f.close()
                        # data = data.replace("'", "\"")
                        #data = data.replace("{", "'{")
                        #data = data.replace("}", "}'")

                        data = ast.literal_eval(data)#change string to a dictionary
                        #print(type(data))

                        for stuff in data:
                            if(stuff == "access_token_startTime"):
                                return data[stuff]
                    
                    except:
                        error = traceback.format_exc()
                        Logging.error(error)


                def Get_refreshToken():
                    try:
                        #turn credentials.json file content into string by loading file into dumps function
                        with open('tokens.dict') as f:
                            data = f.read()
                        f.close()


                        data = ast.literal_eval(data)#change string to a dictionary
                        #print(type(data))

                        for stuff in data:
                            if(stuff == "refresh_token"):
                                return data[stuff]
                    
                    except:
                        error = traceback.format_exc()
                        Logging.error(error)


                def Get_refresh_Token_startTime():

                    try:
                        #turn credentials.json file content into string by loading file into dumps function
                        with open('tokens.dict') as f:
                            data = f.read()
                        f.close()
                        # data = data.replace("'", "\"")
                        #data = data.replace("{", "'{")
                        #data = data.replace("}", "}'")

                        data = ast.literal_eval(data)#change string to a dictionary
                        #print(type(data))

                        for stuff in data:
                            if(stuff == "refresh_token_startTime"):
                                return data[stuff]
                    
                    except:
                        error = traceback.format_exc()
                        Logging.error(error)


                def Get_idToken():

                    try:
                        #turn credentials.json file content into string by loading file into dumps function
                        with open('tokens.dict') as f:
                            data = f.read()
                        f.close()
                        # data = data.replace("'", "\"")
                        #data = data.replace("{", "'{")
                        #data = data.replace("}", "}'")

                        data = ast.literal_eval(data)#change string to a dictionary
                        #print(type(data))

                        for stuff in data:
                            if(stuff == "id_token"):
                                return data[stuff]
                    
                    except:
                        error = traceback.format_exc()
                        Logging.error(error)


                def Refresh_AccessToken(refresh_token):

                    try:
                        #global variables
                        #-----------------------------
                        redirect_url = "https://127.0.0.1"
                        appKey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_appKey()
                        secretKey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_secretKey()
                        # authURL = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={appKey}&redirect_uri={redirect_url}"#    ***** redirect_uri not redirect_url ****
                        # tradingURL = "https://api.schwabapi.com/trader/v1/"        

                        now = datetime.datetime.now()
                        accessToken_startTime = Authentication_Management.Authentication_MyVersion.Get_Authentication_Tokens.Get_accessToken_startTime()
                        

                        #if 28 minutes have passed - refresh the access token

                        #print("current accessToken_StartTime: "+accessToken_startTime)

                        if(now >= (datetime.datetime.strptime(accessToken_startTime,"%Y-%m-%d %H:%M:%S.%f")+datetime.timedelta(minutes=28))):                    


                            headers = {'Authorization': f'Basic {base64.b64encode(bytes(f"{appKey}:{secretKey}", "utf-8")).decode("utf-8")}', 'Content-Type':'application/x-www-form-urlencoded'}
                            data = {'grant_type': 'refresh_token', 'refresh_token': refresh_token}    #    ***** redirect_uri not redirect_url ****

                            
                            response = req.post(url='https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data)
                            
                            if("200" in str(response)):

                                #get refresh token start time

                                now = datetime.datetime.now()

                                refresh_token_startTime = Authentication_Management.Authentication_MyVersion.Get_Authentication_Tokens.Get_refresh_Token_startTime()
                                                    
                                tokenDictionary = response.json()

                                tokenDictionary = str(tokenDictionary).replace("'}", "', 'refresh_token_startTime': '{}', 'access_token_startTime': '{}'}}".format(refresh_token_startTime, now))

                                message = "Updated access_token: "+str(now)

                                # print tokens and response to file
                                f = open("tokens.dict", "w")
                                f.write(str(tokenDictionary))
                                f.close()

                                Logging.info(message)
                                return True
                                            

                                
                            if("200" not in str(response)):

                                Logging.error(str(response)+"\r\f"+response.json())
                                return False
                            
                        return False
                            



                            
                    except:
                        error = traceback.format_exc()
                        Logging.error(error)            


                def Refresh_RefreshToken():#UNFINISHED
                    #global variables
                    #-----------------------------
                    redirect_url = "https://127.0.0.1"
                    appKey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_appKey()
                    secretKey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_secretKey()
                    authURL = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={appKey}&redirect_uri={redirect_url}"#    ***** redirect_uri not redirect_url ****
                    tradingURL = "https://api.schwabapi.com/trader/v1/"        

                    now = datetime.datetime.now()
                    refreshToken_startTime = Authentication_Management.Authentication_MyVersion.Get_Authentication_Tokens.Get_refresh_Token_startTime()
                    

                    if(now < (datetime.datetime.strptime(refreshToken_startTime,"%Y-%m-%d %H:%M:%S.%f")+datetime.timedelta(days=6, hours=23, minutes=58))):

                        timeForMyMark = now + datetime.timedelta(minutes=2) - now
                        Logging.info("You now have "+str(timeForMyMark)+" to refresh the Refresh token")



                        # headers = {'Authorization': f'Basic {base64.b64encode(bytes(f"{appKey}:{secretKey}", "utf-8")).decode("utf-8")}', 'Content-Type':'application/x-www-form-urlencoded'}
                        # data = {'grant_type': 'refresh_token', 'refresh_token=': refresh_token}    #    ***** redirect_uri not redirect_url ****

                        
                        # response = req.post(url='https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data)

                        # if("200" in str(response)):
                    
                        #     message = "Updated the refresh_token - {} - response: {}".format(now, response)
                        #     Logging.info(message)
                            
                        # if("200" not in str(response)):

                        #     Logging.error(str(response))



            def Authenticate_Manually():

                #global variables
                #-----------------------------
                htmlFile = "C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\html_linkFile.html"
                redirect_url = "https://127.0.0.1"
                appKey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_appKey()
                secretKey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_secretKey()
                authURL = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={appKey}&redirect_uri={redirect_url}"#    ***** redirect_uri not redirect_url ****
                tradingURL = "https://api.schwabapi.com/trader/v1/"


                #internal methods
                #-----------------------------

                def CreateHTMLFile(url):

                    # The link CANNOT BE AUTO-OPENED. It must instead be created as a link in a web browser and then clicked by a user.
                    # This is likely a security measure

                    f = open("html_linkFile.html", "a")
                    f.write("<!DOCTYPE html>\r<html>\r<body>\r<a href=\"{}\">Click this link to begin the authentication process.</a></body>\r</html>".format(url))
                    f.close()
                    return
                
                def OpenHTMLFile(htmlFile):
                    webbrowser.open_new_tab(htmlFile)
                
                def DeleteHTMLFile(htmlFile):

                    if(os.path.exists(htmlFile)):
                        os.remove(htmlFile)



                # BEGIN AUTHENTICATE MANUALLY METHOD:

                try:

                    CreateHTMLFile(authURL)
                    OpenHTMLFile(htmlFile)            

                    #Open console to accept the copy/pasted url from the 404 page and get the tokens
                    print(f"The URL below has been activated to begin the manual authentication process.")
                    print()  
                    print()
                    print("After clicking on the url, you will  be prompted to log into your account.")
                    print()
                    print()
                    url404 = input("After you have finished logging in, copy and paste the URL from the 404 page here and then press ENTER: ")
                    DeleteHTMLFile(htmlFile)
                    #authenticationCode = url404[url404.index('code=')+5:url404.index('%40')]#gets the string between code= and %40 in url404

                    authenticationCode = f"{url404[url404.index('code=')+5:url404.index('%40')]}@"

                    # stuff = len(authenticationCode)/4

                    # f = open("code.txt", "w")
                    # f.write(str(authenticationCode)+"             "+str(stuff))
                    # f.write("\f")
                    # f.close()
                    
                    # Get the authorization code from 404 url



                    headers = {'Authorization': f'Basic {base64.b64encode(bytes(f"{appKey}:{secretKey}", "utf-8")).decode("utf-8")}', 'Content-Type':'application/x-www-form-urlencoded'}
                    data = {'grant_type': 'authorization_code', 'code': authenticationCode, 'redirect_uri': redirect_url}    #    ***** redirect_uri not redirect_url ****
                    
                    # f = open("headDataFile.dict", "w")
                    # f.write(str(headers))
                    # f.write("\f")
                    # f.write(str(data))
                    # f.close()
                    
                    
                    response = req.post(url='https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data)

                    # f = open("response.txt", "w")
                    # f.write(str(response))
                    # f.close()



                    tokenDictionary = response.json()

                    #jsonMessage = "\"creation_timestamp\": {}, \"token\": {}\"expires_in\": \"1800\", \"token_type\": \"Bearer\", \"scope\": \"api\", \"access_token\": \"{}\", \"expires_at\": {}}   ".format(Date_Time_Management.Convert_LocalTime_to_EpochTime())

                    # print tokens and response to file
                    f = open("tokens.dict", "w")
                    f.write(str(tokenDictionary))
                    f.close()


                    #add access_token start time
                    #add refresh_token start time
                    #now = datetime.datetime.now()

                    # with open('tokens.dict') as f:
                    #     data = f.read()
                    # f.close()

                    # data = data.replace("'}", "', 'refresh_token_startTime': '{}', 'access_token_startTime': '{}'}}".format(now, now))
                    # f = open("tokens.dict", "w")
                    # f.write(str(data))
                    # f.close()
                    
                    now = Date_Time_Management.Convert_LocalTime_to_EpochTime()
                    then = Date_Time_Management.Convert_LocalTime_to_EpochTime()+1800
                    

                    jsonMessage = "{{\"creation_timestamp\": {}, \"token\": {{\"expires_in\": \"1800\", \"token_type\": \"Bearer\", \"scope\": \"api\", \"refresh_token\": \"{}\", \"access_token\": \"{}\", \"id_token\": \"{}\", \"expires_at\": {} }} }}  ".format(now, Authentication_Management.Authentication_MyVersion.Get_Authentication_Tokens.Get_refreshToken(), Authentication_Management.Authentication_MyVersion.Get_Authentication_Tokens.Get_accessToken(), Authentication_Management.Authentication_MyVersion.Get_Authentication_Tokens.Get_idToken(), then)

                    # print tokens and response to file
                    f = open("tokens.json", "w")
                    f.write(jsonMessage)
                    f.close()

                    if(os.path.exists("tokens.dict")):
                        os.remove("tokens.dict")



                    # # print response to screen
                    # if(response.ok):
                    #     print("response code: "+str(response))
                    #     print()
                    #     print()
                    #     input("Press ENTER when ready.")
                    # if(not response.ok):
                    #     print(str(response))
                    #     print()
                    #     print()
                    #     input("Press ENTER when ready.")
                    # refresh.Access_Token()


                except:
                    error = traceback.format_exc()
                    Logging.error(error)

    class Schwab_Wrapper_Authentication:

        def Get_Authorization_Status():

            auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()

            response = auth.get_account_numbers()

            print(response)


        def Manually_Authenticate():
            appkey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_appKey()
            skey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_secretKey()
            cb_url = "https://127.0.0.1"
            tokenPath = "tokens.json"

            schwab.auth.client_from_manual_flow(appkey, skey, cb_url, tokenPath)

        def Get_Schwab_Account_Authorization():

            #global variables
            #-----------------------------
            redirect_url = "https://127.0.0.1"
            appKey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_appKey()
            secretKey = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_secretKey()
            tokenPath = "tokens.json"


            #try:
            authorization = schwab.auth.easy_client(appKey, secretKey,redirect_url, tokenPath)
               
            return authorization
            
            # except FileNotFoundError:
            #     c = schwab.auth.client_from_manual_flow(
            #         appKey, secretKey, redirect_url, tokenPath)
            # except:
            #     error = traceback.format_exc()            
            #     Logging.error(error)

        def Get_token_expire_time():

            f = open("tokens.json")
            data = json.load(f)

            for headings in data:
                epoch = data['token']['expires_at']

            localTime = Date_Time_Management.Convert_EpochTime_to_LocalTime(epoch)
            return localTime
        

        def Get_token_creation_time():

            f = open("tokens.json")
            data = json.load(f)

            for headings in data:
                epoch = data['creation_timestamp']

            localTime = Date_Time_Management.Convert_EpochTime_to_LocalTime(epoch)
            return localTime


        def Monitor_RefreshToken_Time():

            refreshTokenCreationTime = Authentication_Management.Schwab_Wrapper_Authentication.Get_token_creation_time()
            now = datetime.datetime.now()

            nowWithBufferTime = now+datetime.timedelta(hours=1)

            now = [nowWithBufferTime.date().strftime("%Y-%m-%d"), nowWithBufferTime.time().strftime("%H:%M:%S")]

            print(refreshTokenCreationTime)
            print(now)

            if(Date_Time_Management.Compare_DateTime_Arrays_D1_Before_D2(now, refreshTokenCreationTime)):
                #subprocess.call("Authenticate_Manually.py", shell=True)
                print("Time to authenticate!")
            else:
                print("Keep truckin!")



        class Get_Tokens:

            def Get_access_token():

                f = open("tokens.json")
                data = json.load(f)

                for headings in data:
                    accessToken = data['token']['access_token']

                return accessToken
            
            def Get_refresh_token():

                f = open("tokens.json")
                data = json.load(f)

                for headings in data:
                    refresh_token = data['token']['refresh_token']

                return refresh_token

            def Get_id_token():

                f = open("tokens.json")
                data = json.load(f)

                for headings in data:
                    id_token = data['token']['id_token']

                return id_token


class Calculation_Management:   


    def Calculate_accountBalance_ROI(current_account_balance, previous_account_balance):

        current_account_balance = float(current_account_balance)
        previous_account_balance = float(previous_account_balance)

        roi = (current_account_balance - previous_account_balance)/previous_account_balance

        return roi


    def round(numberToRound, numberOfSigFigs):

        if(numberToRound is None):
            numberToRound = 0.0

        numArray = str(numberToRound).split(".")

        integerside = numArray[0]

        decimalSide = numArray[1]

        lengthOfDecimalSide = len(numArray[1])


        match numberOfSigFigs:

            case 0: #0 - whole number

                if(lengthOfDecimalSide == 0):#no rounding is necessary
                    
                    return numberToRound                
                
                if(lengthOfDecimalSide >= 1):#rounding is necessary
                    
                    if(int(decimalSide[0]) >= 5):
                        
                        numberToRound = int(integerside[0]) + 1

                        #roundedNumber = str(integerside)+"."+str(numberToRound)

                        return float(numberToRound)
                    
                    if(int(decimalSide[0]) < 5):
                        
                        numberToRound = int(integerside)

                        #roundedNumber = str(integerside)+"."+str(numberToRound)

                        return numberToRound

            case 1: #0.0

                if(lengthOfDecimalSide <= 1):#no rounding is necessary
                    
                    return numberToRound                
                
                if(lengthOfDecimalSide > 1):#rounding is necessary
                    
                    if(int(decimalSide[1]) >= 5):
                        
                        numberToRound = int(decimalSide[0]) + 1

                        roundedNumber = str(integerside)+"."+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[1]) < 5):
                        
                        numberToRound = int(decimalSide[0])

                        roundedNumber = str(integerside)+"."+str(numberToRound)

                        return float(roundedNumber)


            case 2: #0.00

                if(lengthOfDecimalSide <= 2):#no rounding is necessary
                    
                    return numberToRound                
              
                if(lengthOfDecimalSide > 2):#rounding is necessary

                    nonsigfigs = str(decimalSide[0])
                    
                    if(int(decimalSide[2]) >= 5):
                        
                        numberToRound = int(decimalSide[1]) + 1

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[2]) < 5):
                        
                        numberToRound = int(decimalSide[1])

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)

            case 3: #0.000

                if(lengthOfDecimalSide <= 3):#no rounding is necessary
                    
                    return numberToRound
                
                if(lengthOfDecimalSide > 3):#rounding is necessary

                    nonsigfigs = str(decimalSide[0]) + str(decimalSide[1])
                    
                    if(int(decimalSide[3]) >= 5):                        
                        
                        numberToRound = int(decimalSide[2]) + 1

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[3]) < 5):
                        
                        numberToRound = int(decimalSide[2])

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
        
            case 4: #0.0000

                if(lengthOfDecimalSide <= 4):#no rounding is necessary
                    
                    return numberToRound
                
                if(lengthOfDecimalSide > 4):#rounding is necessary

                    nonsigfigs = str(decimalSide[0]) + str(decimalSide[1]) + str(decimalSide[2])
                    
                    if(int(decimalSide[4]) >= 5):                        
                        
                        numberToRound = int(decimalSide[3]) + 1

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[4]) < 5):
                        
                        numberToRound = int(decimalSide[3])

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
            case 5: #0.00000

                if(lengthOfDecimalSide <= 5):#no rounding is necessary
                    
                    return numberToRound
                
                if(lengthOfDecimalSide > 5):#rounding is necessary

                    nonsigfigs = str(decimalSide[0]) + str(decimalSide[1]) + str(decimalSide[2]) + str(decimalSide[3])
                    
                    if(int(decimalSide[5]) >= 5):                        
                        
                        numberToRound = int(decimalSide[4]) + 1

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[5]) < 5):
                        
                        numberToRound = int(decimalSide[4])

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
            
            case 6: #0.0000

                if(lengthOfDecimalSide <= 6):#no rounding is necessary
                    
                    return numberToRound
                
                if(lengthOfDecimalSide > 6):#rounding is necessary

                    nonsigfigs = str(decimalSide[0]) + str(decimalSide[1]) + str(decimalSide[2]) + str(decimalSide[3]) + str(decimalSide[4])
                    
                    if(int(decimalSide[6]) >= 5):                        
                        
                        numberToRound = int(decimalSide[5]) + 1

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[6]) < 5):
                        
                        numberToRound = int(decimalSide[5])

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    

            case 7: #0.0000

                if(lengthOfDecimalSide <= 7):#no rounding is necessary
                    
                    return numberToRound
                
                if(lengthOfDecimalSide > 7):#rounding is necessary

                    nonsigfigs = str(decimalSide[0]) + str(decimalSide[1]) + str(decimalSide[2]) + str(decimalSide[3]) + str(decimalSide[4]) + str(decimalSide[5])
                    
                    if(int(decimalSide[7]) >= 5):                        
                        
                        numberToRound = int(decimalSide[6]) + 1

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[7]) < 5):
                        
                        numberToRound = int(decimalSide[6])

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
            case 8: #0.0000

                if(lengthOfDecimalSide <= 8):#no rounding is necessary
                    
                    return numberToRound
                
                if(lengthOfDecimalSide > 8):#rounding is necessary

                    nonsigfigs = str(decimalSide[0]) + str(decimalSide[1]) + str(decimalSide[2]) + str(decimalSide[3]) + str(decimalSide[4]) + str(decimalSide[5]) + str(decimalSide[6])
                    
                    if(int(decimalSide[8]) >= 5):                        
                        
                        numberToRound = int(decimalSide[7]) + 1

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[8]) < 5):
                        
                        numberToRound = int(decimalSide[7])

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
            case 9: #0.0000

                if(lengthOfDecimalSide <= 9):#no rounding is necessary
                    
                    return numberToRound
                
                if(lengthOfDecimalSide > 9):#rounding is necessary

                    nonsigfigs = str(decimalSide[0]) + str(decimalSide[1]) + str(decimalSide[2]) + str(decimalSide[3]) + str(decimalSide[4]) + str(decimalSide[5]) + str(decimalSide[6]) + str(decimalSide[7])
                    
                    if(int(decimalSide[9]) >= 5):                        
                        
                        numberToRound = int(decimalSide[8]) + 1

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[9]) < 5):
                        
                        numberToRound = int(decimalSide[8])

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
            case 10: #0.0000

                if(lengthOfDecimalSide <= 10):#no rounding is necessary
                    
                    return numberToRound
                
                if(lengthOfDecimalSide > 10):#rounding is necessary

                    nonsigfigs = str(decimalSide[0]) + str(decimalSide[1]) + str(decimalSide[2]) + str(decimalSide[3]) + str(decimalSide[4]) + str(decimalSide[5]) + str(decimalSide[6]) + str(decimalSide[7]) + str(decimalSide[8])
                    
                    if(int(decimalSide[10]) >= 5):                        
                        
                        numberToRound = int(decimalSide[9]) + 1

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)
                    
                    if(int(decimalSide[10]) < 5):
                        
                        numberToRound = int(decimalSide[9])

                        roundedNumber = str(integerside)+"."+str(nonsigfigs)+str(numberToRound)

                        return float(roundedNumber)

    def pythonDoesntKnowHowToRound(numberToRound, numberOfSigFigs):

        if(numberOfSigFigs == 1):

            numberOfSigFigs = "0.0"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return float(roundedNumber)

        if(numberOfSigFigs == 2):

            numberOfSigFigs = "0.00"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return roundedNumber

        if(numberOfSigFigs == 3):

            numberOfSigFigs = "0.000"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return float(roundedNumber)

        if(numberOfSigFigs == 4):

            numberOfSigFigs = "0.0000"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return float(roundedNumber)

        if(numberOfSigFigs == 5):

            numberOfSigFigs = "0.00000"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return float(roundedNumber)

        if(numberOfSigFigs == 6):

            numberOfSigFigs = "0.000000"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return float(roundedNumber)

        if(numberOfSigFigs == 7):

            numberOfSigFigs = "0.0000000"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return float(roundedNumber)

        if(numberOfSigFigs == 8):

            numberOfSigFigs = "0.00000000"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return float(roundedNumber)

        if(numberOfSigFigs == 9):

            numberOfSigFigs = "0.000000000"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return float(roundedNumber)

        if(numberOfSigFigs == 10):

            numberOfSigFigs = "0.0000000000"

            roundedNumber =  decimal.Decimal(numberToRound).quantize(decimal.Decimal(numberOfSigFigs), rounding="ROUND_HALF_UP")

            return float(roundedNumber)


    def Calculate_PRR_for_Any_TimePeriod(symbol):

        #  This is the personal rate of return which is
        #  PRR = (amountSold - amountBought)/amountBought over
        #  the current time period it is from the beginning of 
        #  the trading.


        #get current date and time
        cdateTime = datetime.datetime.now()

        currentDateTime = cdateTime.strftime('%Y-%m-%d %H:%M:%S')
        print(currentDateTime)

        #get date and time of the buy-in of the first trade

        firstBuyInDateTime = 0

        #get the sellPrice of the trade
        accountBalanceCurrently = 0

        #get the boughtPrice of the trade

        accountBalanceBeginning = 0

        #prr = (accountBalanceCurrently - accountBalanceBeginning)/accountBalanceBeginning

        #return prr



    def Calculate_HLEVs(symbol, minuteClosePrice, highestMonthly, lowestMonthly, highestWeekly, lowestWeekly):



        #Calculate the hooks law energy variable using the highest high and lowest low in 30 days
        #hlevOrigin float, hlevPercentage float, hlev2000 float, hlev2000_LowerArm float

        hlev2000Constant = 1

        if(minuteClosePrice < 1):
            hlev2000Constant = 0.1
        if(minuteClosePrice > 1 and minuteClosePrice < 10):
            hlev2000Constant = 1
        if(minuteClosePrice > 10 and minuteClosePrice < 100):
            hlev2000Constant = 10
        if(minuteClosePrice > 100 and minuteClosePrice < 1000):
            hlev2000Constant = 100
        if(minuteClosePrice > 1000):
            hlev2000Constant = 1000


        hlev = []


        #calc and store hlevOrigin

        hlevOrigin = lowestMonthly + (highestMonthly - lowestMonthly)/2
        
        #print(hlevOrigin)




        #calc and store hlevPercentage
        
        closePricePosition = minuteClosePrice - hlevOrigin
        denominator = hlevOrigin - lowestMonthly
        



        #calc hlev2000


        hlev2000_middle = lowestWeekly + (highestWeekly - lowestWeekly) / 2


        # hlev2000_Upper = hlev2000_middle + (minuteClosePrice/hlev2000Constant)*0.005* hlev2000_middle
        # hlev2000_Lower = hlev2000_middle - (minuteClosePrice/hlev2000Constant)*0.005* hlev2000_middle

        hlev2000_Upper = hlev2000_middle + (minuteClosePrice/hlev2000Constant)*0.005* hlev2000_middle
        hlev2000_Lower = hlev2000_middle - (minuteClosePrice/hlev2000Constant)*0.005* hlev2000_middle

        
        #print("{} = {} + ({}/{})*0.005*{}".format(hlev2000_Upper,hlev2000_middle,minuteClosePrice,hlev2000Constant,hlev2000_middle))
        hlev.append(Calculation_Management.round(hlevOrigin, 2))
        if(denominator != 0):

            hlevPercentage = closePricePosition/(denominator)
            hlev.append(hlevPercentage)
        if(denominator == 0):
            hlev.append(0.0)
        hlev.append(Calculation_Management.round(hlev2000_Upper,2))
        hlev.append(hlev2000_middle)#hlev.append(Calculation_Management.round(hlev2000_middle,2))
        hlev.append(Calculation_Management.round(hlev2000_Lower,2))


        return hlev


    def Calculate_ProfitLoss_Ratio(symbol, startDate, endDate):
    #    PLR = (totalProfit/numberOfWinningTrades)/(totalProfitLoss/numberOfLosingTrades)
        totalProfit = 0
        totalProfitLoss = 0
        numberOfWinningTrades = 0
        numberOfLosingTrades = 0

        PLR = (totalProfit/numberOfWinningTrades)/(totalProfitLoss/numberOfLosingTrades)
        
        return PLR


    def CalculateAngle(currentEma200Value, previousEma200Value):

        value = currentEma200Value - previousEma200Value

        ema200Angle = math.atan(value * 180 / 3.14159)

        return ema200Angle


    def Calculate_NewestEMA200_Method(previousEMA200Value, newPrice):

        alpha = 2/(200+1)
        ema200Now = previousEMA200Value + alpha*(newPrice-previousEMA200Value)

        return ema200Now

    def Calculate_Newest_StdDev10Avg_Method(previousStdDev10AvgValue, newStdDev10AvgValue):

        # alpha = 2/(200+1)
        sma200Now = (previousStdDev10AvgValue + newStdDev10AvgValue)/2

        return sma200Now

    def Calculate_NewestEMA10Day_Method(previousEMA10DayValue, newPrice):

        #10days = 60mins*6.5 market hours*10 = 3900mins
        alpha = 2/(3900+1)
        ema10DayNow = previousEMA10DayValue + alpha*(newPrice-previousEMA10DayValue)

        return ema10DayNow


    def Calculate_Standard_Dev_of_Angle_of_EMA200(angleArray = []):
        
        
        #sigma = sqrt((difference^2+difference^2+...)/N)
        
        #find the mean
        #

        sumOfAnArray = sum(angleArray)

        mean = sumOfAnArray/200 #nowYourMessinWitha = sumOfAnArray

        differenceArray = []

        for eachAngle in angleArray:

            sqrOfDifference = (eachAngle - mean)**2
            differenceArray.append(sqrOfDifference)

        sumOfDifferences = sum(differenceArray)
        
        #sumOfSquares = sum(squareValueArray)
        average = sumOfDifferences/200
        standardDev_of_Angle = math.sqrt(average)

        return standardDev_of_Angle


    def Calculate_Risk_Amount(buyPrice):
        risk = 0
        profit = 0
        #take from the accountValue your riskAmount

        if (buyPrice < 1.00):
            risk = buyPrice*100
        
        if (buyPrice >= 1.00 and buyPrice < 10.00):
            risk = buyPrice*10

        if (buyPrice > 30.00 and buyPrice <=300.00):
            risk = buyPrice*1 

        return risk


    def Calculate_Share_Amount(buyPrice, accountBalance):
        
        shareAmount = 0
        #baseAllocationPercentage = 0.01  # Default to 1% of account balance

        buyPrice = float(buyPrice)

        coeffecient = accountBalance/buyPrice

        shareAmount = int(coeffecient/10)

        if(shareAmount <=0):
            shareAmount = 1
        
        return shareAmount

        # # Adjust allocation percentage based on ROI:
        # if accountBalance_ROI > 0.20:  # If ROI is above 20%, increase allocation
        #     allocationPercentage = baseAllocationPercentage + 0.02  # Increase allocation to 3%
        # elif accountBalance_ROI > 0.10:  # If ROI is between 10-20%, moderate allocation
        #     allocationPercentage = baseAllocationPercentage + 0.01  # Increase allocation to 2%
        # elif accountBalance_ROI < -0.10:  # If ROI is negative or below -10%, decrease allocation
        #     allocationPercentage = baseAllocationPercentage - 0.005  # Decrease allocation to 0.5%
        # else:
        #     allocationPercentage = baseAllocationPercentage  # Keep the base allocation


        # # Calculate the dollar amount to allocate for the purchase
        # allocationAmount = accountBalance * allocationPercentage
        # #allocationAmount = accountBalance_ROI * allocationPercentage

        # # Calculate the number of shares to buy based on the stock price
        # shareAmount = Calculation_Management.round(allocationAmount / buyPrice, 0)  # Floor division to avoid fractional shares

        # print(shareAmount)
        # print(allocationAmount)
        # print(buyPrice)


        # # For very low-priced stocks, cap the share amount (to avoid massive numbers of shares).
        # maxShares = 100000
        # if shareAmount > maxShares:
        #     shareAmount = maxShares
    
        # return int(shareAmount)

        


        # if(accountBalance <= 5000.00):

        #     if (0 < buyPrice  and buyPrice <= 0.09):
        #         shareAmount = 10000
            
        #     if (0.10 <= buyPrice  and buyPrice <= 0.99):
        #         shareAmount = 1000

        #     if (1.00 <= buyPrice and buyPrice <= 9.99):
        #         shareAmount = 100
            
        #     if (10.00 <= buyPrice and buyPrice <= 99.99):
        #         shareAmount = 10

        #     if (100.00 <= buyPrice and buyPrice <= 499.99):
        #         shareAmount = 1   

        #     return shareAmount
        
        # if(accountBalance > 5000.00 and accountBalance <= 10000.00):

        #     if (buyPrice < 1.00):
        #         shareAmount = 1000

        #     if (1.00 <= buyPrice and buyPrice <= 9.99):
        #         shareAmount = 100
            
        #     if (10.00 <= buyPrice and buyPrice <= 99.99):
        #         shareAmount = 10

        #     if (100.00 <= buyPrice and buyPrice <= 499.99):
        #         shareAmount = 1

        #     if (500.00 <= buyPrice and buyPrice <= 999.99):
        #         shareAmount = 1  

        #     return shareAmount

        # if(accountBalance > 10000.00 and accountBalance <= 25000.00):

        #     if (buyPrice < 1.00):
        #         shareAmount = 1000

        #     if (1.00 <= buyPrice and buyPrice <= 9.99):
        #         shareAmount = 100
            
        #     if (10.00 <= buyPrice and buyPrice <= 99.99):
        #         shareAmount = 10

        #     if (100.00 <= buyPrice and buyPrice <= 499.99):
        #         shareAmount = 5

        #     if (1000.00 <= buyPrice and buyPrice <= 2499.99):
        #         shareAmount = 1  

        #     return shareAmount

        # if(accountBalance > 25000.00 and accountBalance <= 50000.00):

        #     if (buyPrice < 1.00):
        #         shareAmount = 1500

        #     if (1.00 <= buyPrice and buyPrice <= 9.99):
        #         shareAmount = 150
            
        #     if (10.00 <= buyPrice and buyPrice <= 99.99):
        #         shareAmount = 15

        #     if (100.00 <= buyPrice and buyPrice <= 499.99):
        #         shareAmount = 8

        #     if (1000.00 <= buyPrice and buyPrice <= 2499.99):
        #         shareAmount = 3

        #     if (2500.00 <= buyPrice and buyPrice <= 4999.99):
        #         shareAmount = 3  

        #     return shareAmount

        # if(accountBalance > 50000.00 and accountBalance <= 100000.00):

        #     if (buyPrice < 1.00):
        #         shareAmount = 2000

        #     if (1.00 <= buyPrice and buyPrice <= 9.99):
        #         shareAmount = 200
            
        #     if (10.00 <= buyPrice and buyPrice <= 99.99):
        #         shareAmount = 20

        #     if (100.00 <= buyPrice and buyPrice <= 499.99):
        #         shareAmount = 5

        #     if (1000.00 <= buyPrice and buyPrice <= 2499.99):
        #         shareAmount = 5

        #     if (2500.00 <= buyPrice and buyPrice <= 4999.99):
        #         shareAmount = 5 

        #     return shareAmount


        # if(accountBalance > 100000.00 and accountBalance <= 300000.00):

        #     if (buyPrice < 1.00):
        #         shareAmount = 20000

        #     if (1.00 <= buyPrice and buyPrice <= 9.99):
        #         shareAmount = 2000
            
        #     if (10.00 <= buyPrice and buyPrice <= 99.99):
        #         shareAmount = 200

        #     if (100.00 <= buyPrice and buyPrice <= 499.99):
        #         shareAmount = 15

        #     if (1000.00 <= buyPrice and buyPrice <= 2499.99):
        #         shareAmount = 10

        #     if (2500.00 <= buyPrice and buyPrice <= 4999.99):
        #         shareAmount = 8 

            #return shareAmount


    def Calculate_stdDev10Avg(standardDev10Array):

        sma_standardDev10 = (sum(standardDev10Array)/200)

        return abs(sma_standardDev10)


    def Calculate_Upper_EMAs(newest_ClosePrice, newest_EMA200_Value):

        emaArray = []

        hlev2000Constant = 1

        if(newest_ClosePrice < 1):
            hlev2000Constant = 0.1
        if(newest_ClosePrice > 1 and newest_ClosePrice < 10):
            hlev2000Constant = 1
        if(newest_ClosePrice > 10 and newest_ClosePrice < 100):
            hlev2000Constant = 10
        if(newest_ClosePrice > 100 and newest_ClosePrice < 1000):
            hlev2000Constant = 100
        if(newest_ClosePrice > 1000):
            hlev2000Constant = 1000


        resizingConstant = 0.005

        ema1 = newest_EMA200_Value + (newest_ClosePrice/hlev2000Constant) * resizingConstant * newest_EMA200_Value * 2.75
        ema2 = newest_EMA200_Value - (newest_ClosePrice/hlev2000Constant) * resizingConstant * newest_EMA200_Value * 2.75
        #ema3 = newest_EMA200_Value + (newest_ClosePrice/hlev2000Constant) * resizingConstant * newest_EMA200_Value * 0.75

        emaArray.append(ema1)
        emaArray.append(ema2)
        #emaArray.append(ema3)

        return emaArray


class Database_Management:

    def Create_Connection_withoutDatabaseName():

        try:

            lootLoaderPassword = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_LootLoader_Password()

            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password= lootLoaderPassword
                
                )
            
            return mydb
        
        except mysql.connector.Error as err:
            print("Error connecting to MySQL:", err)
            error = traceback.format_exc()
            Logging.error(error)
            return None

    def Create_My_Connection():

        try:

            lootLoaderPassword = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_LootLoader_Password()

            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password= lootLoaderPassword,
                database="lootloaderdatabase",
                port="3306"
                )
            
            return mydb
        
        except mysql.connector.Error as err:
            print("Error connecting to MySQL:", err)
            error = traceback.format_exc()
            Logging.error(error)
            return None
            
    def Create_LLDB_Longterm_Connection():

        try:

            lootLoaderPassword = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_LootLoader_Password()

            mydb = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password= lootLoaderPassword,
            database="lldb_longterm"
            )
            
            return mydb
        except:
                        
            error = traceback.format_exc()
            Logging.error(error)
    
    def Create_TestDB_Connection():

        try:

            lootLoaderPassword = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_LootLoader_Password()

            mydb = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password= lootLoaderPassword,
            database="testdatabase"
            )
            
            return mydb
        except:
                        
            error = traceback.format_exc()
            Logging.error(error)

    def Create_LoggingDB_Connection():
        try:

            lootLoaderPassword = Authentication_Management.Authentication_MyVersion.Get_Keys_From_CredentialsFile.Get_LootLoader_Password()

            mydb = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password= lootLoaderPassword,
            database="logging"
            )
            
            return mydb
        except:
                        
            error = traceback.format_exc()
            Logging.error(error)

    class Create_Delete_Tables:

        class Create_Tables:

            def Create_LootLoaderDatabase_and_All_Necessary_Tables():

                try:

                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("Creating LootLoaderDatabase...")

                    connection = Database_Management.Create_Connection_withoutDatabaseName()
                    mycursor = connection.cursor()

                    mycursor.execute("CREATE DATABASE IF NOT EXISTS lootloaderdatabase;")
                    mycursor.close()
                    print("LootLoaderDatabase created.")
                    print()
                    print("Creating Symbol list...")
                    Database_Management.Create_Delete_Tables.Create_Tables.Create_Symbol_Table_for_All_Symbols()
                    Database_Management.Create_Delete_Tables.Create_Tables.Create_Symbol_Table_for_All_Symbols()
                    print()
                    print("Creating Historical tables for all symbols...")
                    Database_Management.Create_Delete_Tables.Create_Tables.Create_Historical_Tables_ForAll_Symbols()
                    print()
                    print("Creating Trading tables for all symbols...")
                    Database_Management.Create_Delete_Tables.Create_Tables().Create_TradingTables_for_All_Symbols()
                    print()
                    print("All necessary tables have been created in the lootloaderdatabase.")
                    print()

                
                except:
                    error = traceback.format_exc()                                       
                    Logging.error(error)

            def Create_Symbol_Table_for_All_Symbols():
                
                try:

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase","symbols")):

                        with open("Documentation\\symbols.txt") as f:
                            symbols = f.readlines()
                        f.close()

                        #print(symbols)
                        symbolList = []
                        print("Loading symbols...")
                        for symbol in symbols:
                            correct_symbol = symbol.replace("\n","")
                            onlySymbol = re.sub('[^a-zA-Z]+', '', correct_symbol) #regex for returning only the upper and lower case letters
                            
                            symbolList.append(onlySymbol)
                        #print(symbolList)

                        #reset mycursor
                        mycursor.close()
                        connection = Database_Management.Create_My_Connection()
                        mycursor = connection.cursor()


                        print()
                        print("Inserting symbols into symbol table...")
                        print()
                        for symbol in symbolList:
                            symbol = str(symbol).replace("'","")
                            mycursor.execute('INSERT INTO symbols(symbol) VALUES (\'{}\');'.format(symbol))
                            connection.commit()
                        mycursor.close()
                        
                        print("The symbol table has been created in the lootloaderdatabase.")

                        
                    
                    if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase","symbols") == False):

                        connection = Database_Management.Create_My_Connection()
                        mycursor = connection.cursor()
                        mycursor.execute("CREATE TABLE Symbols(Symbol nchar(4));")
                        mycursor.close()
                        print("Symbol table created. Fetching symbols...")

                        with open("Documentation\\symbols.txt") as f:
                            symbols = f.readlines()
                        f.close()

                        #print(symbols)
                        symbolList = []
                        print("Loading symbols...")
                        for symbol in symbols:
                            symbol = str(symbol).replace("'","")
                            correct_symbol = symbol.replace("\n","")
                            onlySymbol = re.sub('[^a-zA-Z]+', '', correct_symbol) #regex for returning only the upper and lower case letters
                            
                            symbolList.append(onlySymbol)
                        #print(symbolList)



                        for symbol in symbolList:
                            symbol = str(symbol).replace("'","")
                            mycursor.execute('INSERT INTO symbols(symbol) VALUES (\'{}\');'.format(symbol))
                            connection.commit()
                        mycursor.close()
                        
                        print("The symbol table has been created in the lootloaderdatabase.")
                except:
                    error = traceback.format_exc()                                       
                    Logging.error(error) 


            def Create_Historical_Tables_ForAll_Symbols():

                try:

                    #get allsymbols and store them in an array

                    allSymbols = []

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    print()
                    print("Fetching Symbol List...")
                    mycursor.execute('SELECT * FROM Symbols')
                    
                    for item in mycursor.fetchall():
                        allSymbols.append(item)

                    #reset cursor    
                    mycursor.close()
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    print("Creating historical tables...")
                    print()
                    tme.sleep(1)
                    for symbol in allSymbols:

                        symbol = str(symbol).replace("'","")
                        symbol = re.sub('[^a-zA-Z]+', '', symbol) #regex for returning only the upper and lower case letters
                        status = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",symbol+"_historical")
                        if( status == False):
                            mycursor.execute('CREATE TABLE {}_Historical(Symbol nchar(4), nDate date, nTime time(0), ClosePrice float, HighPrice float, LowPrice float, Volume float, EMA200 float, EMA200Angle float, stdrdDev1 float, stdrdDev2 float, stdrdDev3 float, stdrdDev4 float, stdrdDev5 float, stdrdDev6 float, stdrdDev7 float, stdrdDev8 float, stdrdDev9 float, stdrdDev10 float, negstdrdDev1 float, negstdrdDev2 float, negstdrdDev3 float, negstdrdDev4 float, negstdrdDev5 float, negstdrdDev6 float, negstdrdDev7 float, negstdrdDev8 float, negstdrdDev9 float, negstdrdDev10 float, stdDev10Avg float, hlevOrigin float, ID int, hlevPercentage float, hlev2000_UpperArm  float, hlev2000_MiddleArm float, hlev2000_LowerArm float);'.format(symbol))
                            print(symbol +" created.")
                        if(status):
                            #time.sleep(0.25)
                            print(symbol+" already exists. Moving to next symbol.")
                            pass
                        
                    
                    print()
                    print("Historical tables have been created in the lootloaderdatabase.")

                except:
                    error = traceback.format_exc()
                    Logging.error(error)


            def Create_Historical_Tables_ForAll_Symbols_LLDBLongterm():

                try:

                    #get allsymbols and store them in an array

                    allSymbols = []

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    print()
                    print("Fetching Symbol List...")
                    mycursor.execute('SELECT * FROM Symbols')
                    
                    for item in mycursor.fetchall():
                        allSymbols.append(item)

                    #reset cursor    
                    mycursor.close()
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    print("Creating historical tables...")
                    print()
                    tme.sleep(1)
                    for symbol in allSymbols:

                        symbol = str(symbol).replace("'","")
                        symbol = re.sub('[^a-zA-Z]+', '', symbol) #regex for returning only the upper and lower case letters
                        status = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",symbol+"_historical")
                        if( status == False):
                            mycursor.execute('CREATE TABLE {}_Historical(Symbol nchar(4), nDate date, nTime time(0), ClosePrice float, HighPrice float, LowPrice float, Volume float, EMA200 float, EMA200Angle float, stdrdDev1 float, stdrdDev2 float, stdrdDev3 float, stdrdDev4 float, stdrdDev5 float, stdrdDev6 float, stdrdDev7 float, stdrdDev8 float, stdrdDev9 float, stdrdDev10 float, negstdrdDev1 float, negstdrdDev2 float, negstdrdDev3 float, negstdrdDev4 float, negstdrdDev5 float, negstdrdDev6 float, negstdrdDev7 float, negstdrdDev8 float, negstdrdDev9 float, negstdrdDev10 float, stdDev10Avg float, hlevOrigin float, ID int, hlevPercentage float, hlev2000_UpperArm  float, hlev2000_MiddleArm float, hlev2000_LowerArm float);'.format(symbol))
                            print(symbol +" created.")
                        if(status):
                            #time.sleep(0.25)
                            print(symbol+" already exists. Moving to next symbol.")
                            pass
                        
                    
                    print()
                    print("Historical tables have been created in the lootloaderdatabase.")

                except:
                    error = traceback.format_exc()
                    Logging.error(error)


            def Create_Historical_Tables_ForAll_Symbols_TestServer():

                try:

                    #get allsymbols and store them in an array

                    allSymbols = []

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    print()
                    print("Fetching Symbol List...")
                    mycursor.execute('SELECT * FROM Symbols')
                    
                    for item in mycursor.fetchall():
                        allSymbols.append(item)

                    #reset cursor    
                    mycursor.close()
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    print("Creating historical tables...")
                    print()
                    tme.sleep(1)
                    for symbol in allSymbols:

                        symbol = str(symbol).replace("'","")
                        symbol = re.sub('[^a-zA-Z]+', '', symbol) #regex for returning only the upper and lower case letters
                        status = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",symbol+"_historical")
                        if( status == False):
                            mycursor.execute('CREATE TABLE {}_Historical(Symbol nchar(4), nDate date, nTime time(0), ClosePrice float, HighPrice float, LowPrice float, Volume float, EMA200 float, EMA200Angle float, stdrdDev1 float, stdrdDev2 float, stdrdDev3 float, stdrdDev4 float, stdrdDev5 float, stdrdDev6 float, stdrdDev7 float, stdrdDev8 float, stdrdDev9 float, stdrdDev10 float, negstdrdDev1 float, negstdrdDev2 float, negstdrdDev3 float, negstdrdDev4 float, negstdrdDev5 float, negstdrdDev6 float, negstdrdDev7 float, negstdrdDev8 float, negstdrdDev9 float, negstdrdDev10 float, stdDev10Avg float, hlevOrigin float, ID int, hlevPercentage float, hlev2000_UpperArm  float, hlev2000_MiddleArm float, hlev2000_LowerArm float);'.format(symbol))
                            print(symbol +" created.")
                        if(status):
                            #time.sleep(0.25)
                            print(symbol+" already exists. Moving to next symbol.")
                            pass
                        
                    
                    print()
                    print("Historical tables have been created in the lootloaderdatabase.")

                except:
                    error = traceback.format_exc()
                    Logging.error(error)


            def Create_Historical_Table_By_Name(tableName):
                
                try:
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    status = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)

                    if(status):
                        message = "The table called {} already exists in the LLDB.\r\rNo table created.".format(tableName)
                        print(message)
                        Logging.info(message)
                        mycursor.close()
                        connection.close()
                        return
                    if(status == False):  
                        print("Creating table...")  
                        mycursor.execute('CREATE TABLE {}(Symbol nchar(4), nDate date, nTime time(0), ClosePrice float, HighPrice float, LowPrice float, Volume float, EMA200 float, EMA200Angle float, stdrdDev1 float, stdrdDev2 float, stdrdDev3 float, stdrdDev4 float, stdrdDev5 float, stdrdDev6 float, stdrdDev7 float, stdrdDev8 float, stdrdDev9 float, stdrdDev10 float, negstdrdDev1 float, negstdrdDev2 float, negstdrdDev3 float, negstdrdDev4 float, negstdrdDev5 float, negstdrdDev6 float, negstdrdDev7 float, negstdrdDev8 float, negstdrdDev9 float, negstdrdDev10 float, stdDev10Avg float, hlevOrigin float, ID int, hlevPercentage float, hlev2000_UpperArm  float, hlev2000_MiddleArm float, hlev2000_LowerArm float);'.format(tableName))
                        message = "The table {} has been created in lootloaderdatabase.".format(tableName)               
                        mycursor.close()
                        connection.close()
   
                except:
                    error = traceback.format_exc()
                    print(error)
                    Logging.error(error)

            
            def Create_Trading_Table_By_Name(tableName):
                
                try:
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    status = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)

                    if(status):
                        message = "The table called {} already exists in the LLDB.\r\rNo table created.".format(tableName)
                        print(message)
                        Logging.info(message)
                        mycursor.close()
                        connection.close()
                        return
                    if(status == False):  
                        print("Creating table...")  
                        mycursor.execute('CREATE TABLE {}(Symbol nchar(4), nDate date, nTime time(0), accountBalance float, lookingForBuy int, AlreadyBought int, lookingForSell int, AlreadySold int,  orderType nchar(4), ClosePrice float, numberOfShares int, Risk float, Profit_Loss float, ROI float, incomeTaxesDue float, buyPrice float, sellPrice float, Good_Bad_Trade nchar(4), order_ID nchar(60), rowID int);'.format(tableName))
                        message = "The table {} has been created in lootloaderdatabase.".format(tableName)               
                        mycursor.close()
                        connection.close()
   
                except:
                    error = traceback.format_exc()
                    print(error)
                    Logging.error(error)

            @staticmethod
            def Create_TradingTables_for_All_Symbols():

                try:

                    #get allsymbols and store them in an array

                    allSymbols = []

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    print()
                    print("Fetching Symbol List...")
                    mycursor.execute('SELECT * FROM Symbols')
                    
                    for item in mycursor.fetchall():
                        allSymbols.append(item)

                    #reset cursor    
                    mycursor.close()
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    print("Creating Trading tables...")
                    print()
                    #time.sleep(1)
                    for symbol in allSymbols:

                        symbol = str(symbol).replace("'","")
                        symbol = re.sub('[^a-zA-Z]+', '', symbol) #regex for returning only the upper and lower case letters
                        status = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",symbol+"_trading")
                        if( status == False):
                            mycursor.execute('CREATE TABLE {}_Trading(Symbol nchar(4), nDate date, nTime time(0), accountBalance float, lookingForBuy int, AlreadyBought int, lookingForSell int, AlreadySold int,  orderType nchar(4), ClosePrice float, numberOfShares int, Risk float, Profit_Loss float, ROI float, incomeTaxesDue float, buyPrice float, sellPrice float, Good_Bad_Trade nchar(4), order_ID nchar(60), rowID int);'.format(symbol))
                            print(symbol +" created.")
                        # if(status):
                        #     #time.sleep(0.25)
                        #     print(symbol+" already exists. Moving to next symbol.")
                        #     pass
                        
                    
                    print()
                    print("Trading tables have been created in the lootloaderdatabase.")

                except:
                    error = traceback.format_exc()
                    Logging.error(error)


            def Create_accountBalance_Table():

                try:
                    
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    
                    print()
                    print("Creating Account_Balance Table...")
                    print()            

                    
                    mycursor.execute('CREATE TABLE Account_Balance(account_balance float, rowID int)')
                    mycursor.close()
                    
                    print()
                    print("Account_Balance table created.")                    
                    print()
                  
                except:
                    error = traceback.format_exc()
                    Logging.error(error)


            def create_logging_table():

                print("are we in...")

                try:
                    print("entef")

                    tday = datetime.datetime.today()
                    todaysDate = tday.date()
                    
                    # connection = Database_Management.Create_LoggingDB_Connection()
                    # mycursor = connection.cursor()
                    print("how about here...")
                    #check if table exists already

                    tableName = "log_"+str(todaysDate).replace("-","")

                    status = True #Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("logging",tableName)
                    print("get status: "+status)

                    if(status == False):

                        print("check status")

                        #create todays table  

                        # mycursor.execute("CREATE TABLE {}(dateTimeStamp DATETIME, message TEXT);".format(tableName))
                        # mycursor.close()
                        print("Created log table for today.")
                    if(status == True):
                        print("no table created")
                        return

                except:
                    error = traceback.format_exc()
                    Logging.error(error)

        class Delete_Tables:

            def Delete_All_Tables_Except_Symbols():
                counter = 1
                try:
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    print()
                    print("Getting all tables in LLDB...")
                    mycursor.execute("SHOW TABLES")
                    print()
                    print("Deleting all tables except for the Symbols table...")
                    for table in mycursor.fetchall():
                        table = str(table).replace("'","")
                        table = str(table).replace(",","")
                        table = str(table).replace("(","")
                        table = str(table).replace(")","")
                        
                        status = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",table)

                        #print("{} - {}".format(table, status))

                        if(status == True and table != "symbols"):
                            mycursor.execute("DROP TABLE {}".format(str(table)))
                            connection.commit()
                            print(counter+" "+table+" deleted.")
                        if(status == False):
                            print(table+" doesn't exist.")
                            pass
                        counter+=1
                    print()
                    print("All tables deleted except for the Symbols table.")
                    print()
                except:
                    error = traceback.format_exc() 
                    print(error)                   
                    Logging.error(error)

            def Delete_All_Historical_Tables():
                pass

            def Delete_All_Trading_Tables():

                try:
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    print()
                    print("Getting all tables in LLDB...")
                    mycursor.execute("SHOW TABLES")
                    print()
                    print("Deleting all Trading tables...")
                    for table in mycursor.fetchall():
                        table = str(table).replace("'","")
                        table = str(table).replace(",","")
                        table = str(table).replace("(","")
                        table = str(table).replace(")","")
                        
                        
                        status = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",table)


                        if(status == True and "trading" in table):
                            mycursor.execute("DROP TABLE {}".format(str(table)))
                            connection.commit()
                            print(table+" deleted.")
                        if(status == False):
                            print(table+" doesn't exist.")
                            pass
                    print()
                    print("All tables Trading tables deleted.")
                    print()
                except:
                    error = traceback.format_exc() 
                    print(error)                   
                    Logging.error(error)

            def Delete_Table_By_Name(tableNameToDelete):

                try:

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("DROP TABLE lootloaderdatabase.{};".format(tableNameToDelete))
                    connection.commit()

                    print("{} has been deleted from lootloaderdatabase".format(tableNameToDelete))
                    Logging.info("{} has been deleted from lootloaderdatabase".format(tableNameToDelete))

                except:
                    error = traceback.format_exc()
                    print(error)                    
                    Logging.error(error)

            def Purge_SymbolsTable_of_LowData_Symbols(symbol, counter_ofSymbols, numberOfSymbols):
                #only use this method on new tables
                try:

                    tableName = "{}_historical".format(symbol)

                    recordCount = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Total_Amount_of_Records_InTable(tableName)

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    message = "{} of {}. {} - Backfilled to present with {} records stored.".format(counter_ofSymbols, numberOfSymbols,symbol, recordCount)+"\r\f"
                    print(message)
                            
                    Logging.info(message)

                    

                    if(recordCount < 6000):

                        mycursor.execute("DROP TABLE lootloaderdatabase.{};".format(tableName))
                        connection.commit()
                        tableName = "{}_trading".format(symbol)
                        mycursor.execute("DROP TABLE lootloaderdatabase.{};".format(tableName))
                        connection.commit()
                        mycursor.execute("DELETE FROM symbols WHERE symbol=\'{}\'".format(symbol))
                        # connection.commit()
                        message = "{} has {} records after initial install and therefore has been deleted from lootloaderdatabase and the symbols table.".format(symbol, recordCount)
                        print(message)
                        Logging.info(message)

                        mycursor.close()
                        return "No trade"
                    return "Good to trade"
                    # connection.close()

                    

                except:
                    error = traceback.format_exc()
                    print(error)                    
                    Logging.error(error)

        def Reload_the_Symbols_Table():
            try:

                connection = Database_Management.Create_My_Connection()
                mycursor = connection.cursor()

                mycursor.execute("DROP TABLE lootloaderdatabase.symbols;")
                connection.commit()

                Database_Management.Create_Delete_Tables.Create_Tables.Create_Symbol_Table_for_All_Symbols()


                message = "The symbols table has been reloaded from its original status."
                Logging.info()

            except:
                error = traceback.format_exc()
                print(error)                
                Logging.error(error)


    class Get_Data_From_LootLoaderDataBase:

        class Historical:

            def Get_Symbol_List():

                symbolList = []

                try:
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("SELECT * FROM lootloaderdatabase.symbols")
                    lst = mycursor.fetchall() #without this mycursor will return nothing
                    for symbol in lst:
                        symbol = str(symbol).replace("('","")
                        symbol = str(symbol).replace("',)","")
                        symbolList.append(symbol)
                    return symbolList
                
                except:
                    error = traceback.format_exc()
                    Logging.error(error)

            def Get_Sectioned_Symbol_List_85():

                try:

                    counter = 1

                    symbolList = []
                    sectionedSymbolList = []

                    print("Getting list..")
                    print()

                    entireSymbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()

                    print("Making sectioned lists..")
                    print()

                    for symbol in entireSymbolList:

                        symbolList.append(symbol)

                        if(counter == 85):

                            sectionedSymbolList.append(symbolList)

                            symbolList = []

                            counter = 1

                        counter += 1

                    print("Done.")
                    print()

                    return sectionedSymbolList
                
                    

                except:
                    error = traceback.format_exc()
                    Logging.error(error)      


            def Get_Last_Row_of_Historical_Tables(symbol):

                tableName = "{}_historical".format(symbol)

                if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)):
                    status = ""
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    #insert data int table
                    mycursor.execute('SELECT * FROM lootloaderdatabase.{}_Historical ORDER BY ID desc LIMIT 1;'.format(symbol))
                    ihatePython = mycursor.fetchone()# in order to produce a "None" which describes an empty record you have to do this because pyodbc doesn't handle null records
                    #cursor.close()

                    if(ihatePython is None):


                        tday = datetime.datetime.today()
                        todaysDate = tday.date()
                        now = tday.time()
                        print("There are no records present for: "+symbol)
                        

                        message = "No records present."
                        Logging.info(message)

                        return(None)
                        
                    else:

                        mycursor.execute('SELECT * FROM lootloaderdatabase.{}_Historical ORDER BY ID desc LIMIT 1;'.format(symbol))
                        for i in mycursor:

                            # st = str(i)
                            # status = st.replace("(","")
                            # status = status.replace(")","")
                            # status = status.replace("'","")
                            # status = status.replace("\"","")
                            # row = status.split(",")
                            #print(status)
                            return i

                        mycursor.close()

                        #return row


            def Get_Latest_ClosePrice(symbol):

                try:
                    lastRow = Database_Management.Get_Data_From_LootLoaderDataBase.Get_Last_Row_of_Historical_Tables(symbol)

                    if(lastRow is None):
                        return 0.0

                    if(lastRow is not None):
                    
                        latestClosePrice = lastRow[3]
                        return latestClosePrice
                
                    
                except:
                    error = traceback.format_exc()                                       
                    Logging.error(error)


            def Get_Row_By_Index(symbol, indx):    
                            
                connection = Database_Management.Create_My_Connection()
                mycursor = connection.cursor()

                #insert data int table
                mycursor.execute('SELECT * FROM {}_historical WHERE ID = {};'.format(symbol, indx))

                for i in mycursor:

                    

                    return i

                mycursor.close()
                
                
            def Get_Previous_200_EMA200Angles_from_DB(symbol):

                lastRow =  Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Last_Row_of_Historical_Tables(symbol)
                lastrowIndex = int(lastRow[31])
                stdDevArray = []
                bottom = lastrowIndex - 199
        
                #print(str(lastrowIndex))
                
                while(lastrowIndex >= bottom):
                    
                    if(bottom < 0):
                        
                        bottom = 1
                    
                    stdDev = float(Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, lastrowIndex)[8])
                    
                    if(stdDev is None or stdDev == "" or stdDev == "None"):

                        
                        return
                    
                    stdDevArray.append(stdDev)
                    

                    lastrowIndex = lastrowIndex - 1

                return stdDevArray


            def Get_Previous_200_StdDev10Values_from_DB(symbol):

                lastRow =  Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Last_Row_of_Historical_Tables(symbol)
                lastrowIndex = int(lastRow[31])
                stdDev10avgArray = []
                bottom = lastrowIndex - 199
        
                #print(str(lastrowIndex))
                
                while(lastrowIndex >= bottom):

                    if(bottom < 0):
                        bottom = 1
                    
                    stdDev = float(Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, lastrowIndex)[18])
                    #print(rowIndex)
                    if(stdDev is None or stdDev == ""):
                        return
                    stdDev10avgArray.append(stdDev)

                    lastrowIndex = lastrowIndex - 1

                return stdDev10avgArray

            def Get_Total_Amount_of_Records_InTable(tableName):

                try:
                    # if(isThisANewTable == False):
                    #    return

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("SELECT COUNT(*) FROM {};".format(tableName))

                    rowAmount = str(mycursor.fetchall())
                    rowAmount = rowAmount.replace("[","")
                    rowAmount = rowAmount.replace("]","")
                    rowAmount = rowAmount.replace("(","")
                    rowAmount = rowAmount.replace(")","")
                    rowAmount = rowAmount.replace(",","")
                    mycursor.close()
                    return int(rowAmount)

                
                    
                except:
                    error = traceback.format_exc()                                       
                    Logging.error(error)


            def Get_the_Highest_MonthlyHigh_From_DataBase(symbol, nDate):

                try:

                    dateArray = []
                    #parse todaysDate
                    dateArray = str(nDate).split("-")

                    nDate = datetime.date(int(dateArray[0]), int(dateArray[1]), int(dateArray[2]))

                    previousDate = nDate - datetime.timedelta(days=30)
                    nTime = "14:59"

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("SELECT MAX(ClosePrice) FROM lootloaderdatabase.{}_historical WHERE(nDate BETWEEN '{}' AND '{}') AND nTime='{}' LIMIT 1".format(symbol,previousDate,nDate, nTime))

                    HighPrice = str(mycursor.fetchall())
                    HighPrice = HighPrice.replace("[","")
                    HighPrice = HighPrice.replace("]","")
                    HighPrice = HighPrice.replace("(","")
                    HighPrice = HighPrice.replace(")","")
                    HighPrice = HighPrice.replace(",","")
                    HighPrice = HighPrice.replace(" ","")
                    mycursor.close()


                    #print("'"+HighPrice+"'")
                    #print(type(HighPrice))

                    if(HighPrice == "None"):
                        return float("0.00")

                    if(HighPrice != "None"):
                        #print(HighPrice)
                        return float(HighPrice)

                
                    
                except:
                    error = traceback.format_exc()                                       
                    Logging.error(error)


            def Get_the_Lowest_MonthlyLowh_From_DataBase(symbol, nDate):
                    
                try:
                    dateArray = []
                    #parse todaysDate
                    dateArray = str(nDate).split("-")

                    nDate = datetime.date(int(dateArray[0]), int(dateArray[1]), int(dateArray[2]))

                    previousDate = nDate - datetime.timedelta(days=30)
                    nTime = "14:59"

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("SELECT MIN(ClosePrice) FROM lootloaderdatabase.{}_historical WHERE(nDate BETWEEN '{}' AND '{}') AND nTime='{}' LIMIT 1".format(symbol,previousDate,nDate, nTime))

                    LowPrice = str(mycursor.fetchall())
                    LowPrice = LowPrice.replace("[","")
                    LowPrice = LowPrice.replace("]","")
                    LowPrice = LowPrice.replace("(","")
                    LowPrice = LowPrice.replace(")","")
                    LowPrice = LowPrice.replace(",","")
                    LowPrice = LowPrice.replace(" ","")
                    mycursor.close()


                    #print("'"+HighPrice+"'")
                    #print(type(HighPrice))

                    if(LowPrice == "None"):
                        return float("0.00")

                    if(LowPrice != "None"):
                        #print(LowPrice)
                        return float(LowPrice)

                
                    
                except:
                    error = traceback.format_exc()
                    print(error)                    
                    Logging.error(error)


            def Get_the_Highest_WeeklyHigh(symbol, todaysDate):

                try:
                    dateArray = []
                    #parse todaysDate
                    dateArray = str(todaysDate).split("-")

                    todaysDate = datetime.date(int(dateArray[0]), int(dateArray[1]), int(dateArray[2]))

                    previousDate = todaysDate - datetime.timedelta(minutes=2500)#5.1282days = 2000hours

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("SELECT MAX(ClosePrice) FROM lootloaderdatabase.{}_historical WHERE(nDate BETWEEN '{}' AND '{}')".format(symbol,previousDate,todaysDate))

                    HighPrice = str(mycursor.fetchall())
                    HighPrice = HighPrice.replace("[","")
                    HighPrice = HighPrice.replace("]","")
                    HighPrice = HighPrice.replace("(","")
                    HighPrice = HighPrice.replace(")","")
                    HighPrice = HighPrice.replace(",","")
                    mycursor.close()

                    #print(HighPrice)

                    if(HighPrice == "None"):
                        return float("0.00")

                    if(HighPrice != "None"):
                        return float(HighPrice)

                
                    
                except:
                    error = traceback.format_exc()
                    print(error)                    
                    Logging.error(error)


            def Get_the_Lowest_WeeklyLow(symbol, todaysDate):

                try:
                    dateArray = []
                    #parse todaysDate
                    dateArray = str(todaysDate).split("-")

                    todaysDate = datetime.date(int(dateArray[0]), int(dateArray[1]), int(dateArray[2]))

                    previousDate = todaysDate - datetime.timedelta(minutes=2500)#5.1282days = 2000hours

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("SELECT MIN(ClosePrice) FROM lootloaderdatabase.{}_historical WHERE(nDate BETWEEN '{}' AND '{}')".format(symbol,previousDate,todaysDate))

                    HighPrice = str(mycursor.fetchall())
                    HighPrice = HighPrice.replace("[","")
                    HighPrice = HighPrice.replace("]","")
                    HighPrice = HighPrice.replace("(","")
                    HighPrice = HighPrice.replace(")","")
                    HighPrice = HighPrice.replace(",","")
                    mycursor.close()


                    if(HighPrice == "None"):
                        return float("0.00")

                    if(HighPrice != "None"):
                        return float(HighPrice)

                
                    
                except:
                    error = traceback.format_exc()
                    print(error)                    
                    Logging.error(error)



        class Trading:

            
            def Get_Last_Row_of_Trading_Tables(symbol):

                try:

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    symbolStatus = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase","{}_trading".format(symbol))
                    if(symbolStatus == False):
                        Database_Management.Delete_Data_From_LootLoaderDataBase.Historical.Delete_Symbol_from_Symbol_Table(symbol)
                        return
                    mycursor.execute('SELECT * FROM {}_Trading ORDER BY order_ID desc LIMIT 1;'.format(symbol))
                    ihatePython = mycursor.fetchone()# in order to produce a "None" which describes an empty record you have to do this because pyodbc doesn't handle null records
                    # #cursor.close()
                    #print(ihatePython)

                    if(ihatePython is None):
                        return "None"

                    else:
                        mycursor.execute('SELECT * FROM {}_Trading ORDER BY order_ID desc LIMIT 1;'.format(symbol))
                        for i in mycursor:

                        # st = str(i)
                        # status = st.replace("(","")
                        # status = status.replace(")","")
                        # status = status.replace("'","")
                        # status = status.replace("\"","")
                        # row = status.split(",")

                            return i


                    #else:
                    #return(None)
                        
            
                    

                    mycursor.close()

            
                    
                except:
                    error = traceback.format_exc()                                       
                    Logging.error(error)

            def Get_buyPrice_from_DB(symbol):

                try:
                    lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)

                    if(lastTradingRow is None):
                        return 0.0

                    if(lastTradingRow is not None):
                    
                        buyPrice = lastTradingRow[15]
                        return buyPrice
                
                    
                except:
                    error = traceback.format_exc()                                       
                    Logging.error(error)
           
            def Get_orderID_from_DB(symbol):

                try:
                    lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)

                    if(lastTradingRow is None):
                        return 0.0

                    if(lastTradingRow is not None):
                    
                        orderID = lastTradingRow[18]
                        return orderID
                
                    
                except:
                    error = traceback.format_exc()                                       
                    Logging.error(error)

            def Get_numberOfShares_from_DB(symbol):

                try:
                    lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)

                    if(lastTradingRow is None):
                        return 0.0

                    if(lastTradingRow is not None):
                    
                        numberOfShares = lastTradingRow[10]
                        return numberOfShares
                
                    
                except:
                    error = traceback.format_exc()                                       
                    Logging.error(error)


    class Delete_Data_From_LootLoaderDataBase:

        class Historical:

            

            def Delete_All_Symbols_in_SymbolsTable():
                try:
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("DELETE FROM lootloaderdatabase.symbols;")
                    connection.commit() 

                    print("All symbols have been deleted from symbols table.")
                    message = "All symbols have been deleted from symbols table."
                    Logging.info(message)

                    mycursor.close()   

                except:

                    error = traceback.format_exc()
                    Logging.error(error)  

            def Delete_All_HistoricalData_For_OneSymbol(symbol):

                try:
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("TRUNCATE {}_Historical".format(symbol)) 

                    print("All historical data deleted for {}".format(symbol))
                    message = "All historical data deleted for {}".format(symbol)
                    Logging.info(message)

                    mycursor.close()   

                except:

                    error = traceback.format_exc()
                    Logging.error(error)

            def Delete_All_HistoricalData_For_All_Symbols():
                os.system("CLS")
                try:

                    symbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    counter = 1
                    for symbol in symbolList:

                        if(mycursor == None):
                            message = "All historical data has been deleted for all symbols"
                            print(message)
                            Logging.info(message)
                        if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase","{}_historical".format(symbol))):
                            mycursor.execute("TRUNCATE {}_historical".format(symbol)) 

                            message = str(counter)+" All historical data deleted for {}".format(symbol)
                            print(message)
                            Logging.info(message)
                        counter+=1
                    mycursor.close()   

                except:

                    error = traceback.format_exc()
                    Logging.error(error)

            def Delete_Symbol_from_Symbol_Table(symbol):
                try:

                    symbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    message = "Deleting {} from Symbols table...".format(symbol)
                    print(message)
                    print()
                    Logging.info(message)

                    for symbolName in symbolList:

                        if(symbol == symbolName):

                            mycursor.execute("DELETE FROM Symbols WHERE Symbol=\'{}\'".format(symbol))
                            connection.commit()

                    message = "{} has been deleted from the Symbols table.".format(symbol)
                    print(message)
                    print()
                    Logging.info(message)

                    mycursor.close()   

                except:

                    error = traceback.format_exc()
                    Logging.error(error)

            def Delete_Todays_Data_Only_for_Symbol(symbol):
                tday = datetime.datetime.today()
                todaysDate = tday.date()
                try:

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("DELETE FROM {}_historical WHERE nDate = \'{}\'".format(symbol, str(todaysDate)))
                    connection.commit()
                    mycursor.close()   

                except:

                    error = traceback.format_exc()
                    Logging.error(error)
  
            def Delete_Yesterdays_and_Todays_Data_Only_for_Symbol(symbol):
                tday = datetime.datetime.today()
                todaysDate = tday.date()
                yesterdaysDate = tday.date() - datetime.timedelta(days=1)

                try:

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("DELETE FROM {}_historical WHERE nDate >= \'{}\'".format(symbol, str(yesterdaysDate)))
                    connection.commit()
                    mycursor.close()   

                except:

                    error = traceback.format_exc()
                    Logging.error(error)
 
            def Delete_Symbol_if_HistoricalTable_DoesntExist(symbol):

                try:

                    tableName = "{}_Historical".format(symbol)

                    status = Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)

                    if(status == False):

                        tableName = "{}_Trading".format(symbol)

                        Database_Management.Delete_Data_From_LootLoaderDataBase.Historical.Delete_Symbol_from_Symbol_Table(symbol)
                        Database_Management.Create_Delete_Tables.Delete_Tables.Delete_Table_By_Name(tableName)


                except:

                    error = traceback.format_exc()
                    Logging.error(error)
 
        class Trading:

            
            def Delete_All_TradeData_For_All_Symbols():
                os.system("CLS")
                try:

                    symbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()

                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
                    counter = 1
                    for symbol in symbolList:

                        if(mycursor == None):
                            message = "All Trade data has been deleted for all symbols"
                            print(message)
                            Logging.info(message)
                        if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase","{}_Trading".format(symbol))):
                            mycursor.execute("TRUNCATE {}_Trading".format(symbol)) 

                            message = str(counter)+" All Trade data deleted for {}".format(symbol)
                            print(message)
                            Logging.info(message)
                        counter+=1
                    mycursor.close()   

                except:

                    error = traceback.format_exc()
                    Logging.error(error)


            def Delete_All_Trade_Data_for_One_Symbol(symbol):
                try:
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()

                    mycursor.execute("TRUNCATE {}_Trading".format(symbol)) 

                    print("All Trade data deleted for {}".format(symbol))
                    message = "All Trade data deleted for {}".format(symbol)
                    Logging.info(message)

                    mycursor.close()   

                except:

                    error = traceback.format_exc()
                    Logging.error(error)


            def Delete_the_Last_Trading_Row_For_Symbol(symbol):

                os.system("CLS")
                try:
                    connection = Database_Management.Create_My_Connection()
                    mycursor = connection.cursor()
      
                    if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase","{}_trading".format(symbol))):
                        mycursor.execute("DELETE FROM {}_trading order by rowID desc limit 1".format(symbol)) 
                        connection.commit()

                        message = "The last line has been deleted from {}_Trading table.".format(symbol)
                        print(message)
                        Logging.info(message)
                    mycursor.close()   

                except:

                    error = traceback.format_exc()
                    Logging.error(error)                


    class Give_Data_To_LootLoaderDataBase:

        def Give_Symbols_To_DB_SymbolsTable():

            try:

                with open("Documentation\\symbols.txt") as f:
                    symbols = f.readlines()
                f.close()

                #print(symbols)
                symbolList = []

                for symbol in symbols:
                    correct_symbol = symbol.replace("\n","")
                    onlySymbol = re.sub('[^a-zA-Z]+', '', correct_symbol) #regex for returning only the upper and lower case letters
                    
                    symbolList.append(onlySymbol)
                #print(symbolList)
                connection = Database_Management.Create_My_Connection()
                mycursor = connection.cursor()

                for symbol in symbolList:
                    mycursor = connection.cursor()# this has to be in the for loop to work
                    mycursor.execute('INSERT INTO symbols(symbol) VALUES (\'{}\');'.format(symbol))
                    connection.commit()
                mycursor.close()
                
                print("The symbol table has been created in the lootloaderdatabase.")
            except:
                error = traceback.format_exc()                                     
                
                Logging.error(error) 

        def Give_HistoricalData_To_DB(symbol, localDate, localTime, closePrice, highPrice, lowPrice, volume, newEma200Value, newEMA200AngleValue, 
                                        standardDev, standardDev2, standardDev3, standardDev4, standardDev5, standardDev6, standardDev7, 
                                        standardDev8, standardDev9, standardDev10, negStandardDev, negStandardDev2, negStandardDev3, 
                                        negStandardDev4, negStandardDev5, negStandardDev6, negStandardDev7, negStandardDev8, negStandardDev9,
                                        negStandardDev10, stdDev10Avg, hlevOrigin,rowID, hlevPercentage, hlev2000_UpperArm, hlev2000_MiddleArm, hlev2000_LowerArm):

            connection = Database_Management.Create_My_Connection()
            mycursor = connection.cursor()
            

            mycursor.execute('INSERT INTO {}_Historical VALUES (\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\')'.format(symbol, symbol, localDate, localTime, 
                                                                                                                   closePrice, highPrice, lowPrice, volume, 
                                                                                                                   newEma200Value, newEMA200AngleValue, standardDev, standardDev2, 
                                                                                                                   standardDev3, standardDev4, standardDev5, standardDev6, 
                                                                                                                   standardDev7, standardDev8, standardDev9, standardDev10, 
                                                                                                                   negStandardDev, negStandardDev2, negStandardDev3,
                                                                                                                   negStandardDev4, negStandardDev5, negStandardDev6, negStandardDev7, 
                                                                                                                   negStandardDev8, negStandardDev9, negStandardDev10, stdDev10Avg, 
                                                                                                                   hlevOrigin, rowID, hlevPercentage, hlev2000_UpperArm, 
                                                                                                                   hlev2000_MiddleArm, hlev2000_LowerArm))
            
            connection.commit()
            mycursor.close()

        def Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                lookingForBuy, alreadyBought, lookingForSell, 
                                alreadySold, orderType, closePrice, numberOfShares, 
                                riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                orderID, rowID):
            
            try:

                connection = Database_Management.Create_My_Connection()
                mycursor = connection.cursor()
                

                mycursor.execute('INSERT INTO {}_trading VALUES (\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\')'.format(symbol, symbol, localDate, localTime, accountBalance, 
                                                                                                    lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                    alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                    riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                    orderID, rowID))
                
                connection.commit()
                mycursor.close()

            except:
                error = traceback.format_exc()                                       
                Logging.error(error)

        def Give_accountBalance_to_DB(accountBalance, rowID):

            try:

                connection = Database_Management.Create_My_Connection()
                mycursor = connection.cursor()
                

                mycursor.execute('INSERT INTO lootloaderdatabase.account_balance VALUES (\'{}\',\'{}\')'.format(accountBalance, rowID))
                
                connection.commit()
                mycursor.close()

            except:
                error = traceback.format_exc()                                       
                Logging.error(error)

        class Trading:

            def Give_LookingForBuy_Data_to_Symbol():
                pass

            def Give_Buy_Data_to_Symbol():
                pass

            def Give_Sell_Data_to_Symbol(symbol):

                tday = datetime.datetime.today()
                todaysDate = str(tday.date())
                now = tday.time()


                
                    
                closePrice = 0

                buyPrice = 0

                dBdate = todaysDate
                time = now



    class Check_LootLoaderDataBase:

        def Check_If_Table_Exists_in_LLDB(databaseName, tableName):

            try: 

                if(databaseName == "lldb_longterm"): 

                    connection = Database_Management.Create_LLDB_Longterm_Connection()

                if(databaseName == "logging"): 

                    connection = Database_Management.Create_LoggingDB_Connection()  

                if(databaseName == "lootloaderdatabase"): 

                    connection = Database_Management.Create_My_Connection() 

                if(databaseName == "testdatabase"): 

                    connection = Database_Management.Create_TestDB_Connection()          

                
                mycursor = connection.cursor()

                mycursor.execute("SHOW TABLES LIKE \'{}\';".format(tableName))

                row = str(mycursor.fetchall())

                #print(row)

                if(row != "[]"):
                    #print(1)
                    return True
                if(row == "[]"):
                    #print(2)
                    return False
                mycursor.close()
            except:
                error = traceback.format_exc()                                       
                Logging.error(error)

        def Check_If_RecordAmount_is_Enough(symbol):

            try:
                tableName = "{}_historical".format(symbol)

                totalRows = Database_Management.Get_Data_From_LootLoaderDataBase.Get_Total_Amount_of_Records_InTable(tableName)

                connection = Database_Management.Create_My_Connection()
                mycursor = connection.cursor()

                if(totalRows < 10000):
                    mycursor.execute("DROP TABLE {};".format(tableName))
                    message = "Table {} has been deleted because it had less than 10,000 records on a first install.".format(tableName)
                    print(message)
                    Logging.info(message)                
                
                mycursor.close()

            except:
                error = traceback.format_exc()                                       
                Logging.error(error)


class Date_Time_Management:

    def Get_Todays_Weekday_Name():

        try:

            todaysNumberAccordingToPython = dt.today().weekday()

            if(todaysNumberAccordingToPython == 0):
                return str("Monday")
            elif(todaysNumberAccordingToPython == 1):
                return str("Tuesday")
            elif(todaysNumberAccordingToPython == 2):
                return str("Wednesday")
            elif(todaysNumberAccordingToPython == 3):
                return str("Thursday")
            elif(todaysNumberAccordingToPython == 4):
                return str("Friday")
            elif(todaysNumberAccordingToPython == 5):
                return str("Saturday")
            elif(todaysNumberAccordingToPython == 6):
                return str("Sunday")
            
        except:
            error = traceback.format_exc()            
            Logging.error(error)
  
    def Convert_LocalTime_to_EpochTime():
        tday = datetime.datetime.today().replace(microsecond=0)
        pattern = '%Y-%m-%d %H:%M:%S'
        epoch = int(tme.mktime(tme.strptime(str(tday), pattern)))
        # print(tday)
        # print(epoch)
        return epoch
    
    def Convert_EpochTime_to_LocalTime(epochTime):


        localDate = datetime.datetime.fromtimestamp(int(str(epochTime))).strftime('%Y-%m-%d')
        localTime = datetime.datetime.fromtimestamp(int(str(epochTime))).strftime('%X')

        dateTime = [localDate, localTime]

        return dateTime

    def Convert_GivenDateTime_to_EpochTime(DB_DateTime):

        DB_DateTime_string = "{} {}".format(DB_DateTime[0], DB_DateTime[1])
        
        epoch_time = int(tme.mktime(tme.strptime(DB_DateTime_string, "%Y-%m-%d %H:%M:%S")))

        return epoch_time

    def Compare_Epoch_DateTime_D1_GreaterThan_D2(epoch1, epoch2):

        try:
            if(epoch1 > epoch2):
                return True
            if(epoch1 <= epoch2):
                return False
        except:
            error = traceback.format_exc()
            Logging.error(error)

    def Compare_DateTime_Arrays_D1_Before_D2(dateTimeList1, dateTimeList2):

        #this method will compare two datetime arrays and 
        #return true if dateTimeList1 is before dateTimeList2
        #each dateTimeList must be in YYYY-MM-DD HH:MM:SS format
        #or YYYYMMDD HH:MM:SS format

        #charles schwab gives 00 for seconds in minute based data. The .join doesn't like this because
        #when you join two strings that are "0" it will only return one 0 and not two. And so the time#[5] doesn't have anything in it.
        #So I took out the seconds

        try:
            if(dateTimeList1==None or dateTimeList2==None):
                input("Please enter two datetime lists for comparison.")



            date1 = list(str(dateTimeList1[0]).replace("-",""))
            date2 = list(str(dateTimeList2[0]).replace("-",""))
            time1 = list(str(dateTimeList1[1]).replace(":",""))
            time2 = list(str(dateTimeList2[1]).replace(":",""))





            year1 = int(''.join([date1[0],date1[1],date1[2],date1[3]]))
            month1 = int(''.join([date1[4],date1[5]]))
            day1 = int(''.join([date1[6],date1[7]]))

            hour1 = int(''.join([time1[0],time1[1]]))
            minute1 = int(''.join([time1[2],time1[3]]))
            #second1 = int(''.join([time1[4],time1[5]]))


            year2 = int(''.join([date2[0],date2[1],date2[2],date2[3]]))
            month2 = int(''.join([date2[4],date2[5]]))
            day2 = int(''.join([date2[6],date2[7]]))

            hour2 = int(''.join([time2[0],time2[1]]))
            minute2 = int(''.join([time2[2],time2[3]]))
            #second2 = int(''.join([time2[4],time2[5]]))





            dateTime1 = datetime.datetime(year1,month1,day1, hour1, minute1)#, second1)
            dateTime2 = datetime.datetime(year2,month2,day2, hour2, minute2)#, second2)

            # print(dateTime1)
            # print(dateTime2)


            if(dateTime1 < dateTime2):
                return True
            else:
                return False
            
        except:
            error = traceback.format_exc()
            Logging.error(error)


class Interface_Management:

    def Get_total_profit_gain_per_symbol(symbol):

        try:

            connection = Database_Management.Create_My_Connection()
            mycursor = connection.cursor()
            #insert data int table
            mycursor.execute('SELECT Profit_Loss FROM {}_trading WHERE Good_Bad_Trade=\'GOOD\';'.format(symbol))
            ihatePython = mycursor.fetchone()# in order to produce a "None" which describes an empty record you have to do this because pyodbc doesn't handle null records
            # #cursor.close()
            #print(ihatePython)

            if(ihatePython is None):
                return "None"

            else:
                mycursor.execute('SELECT Profit_Loss FROM {}_trading WHERE Good_Bad_Trade=\'GOOD\';'.format(symbol))
                for i in mycursor:
                    i = str(i).replace("(","")
                    i = str(i).replace(",","")
                    i = str(i).replace(")","")

                    return float(i)
      

            mycursor.close()


        
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)

    
    def Get_total_profit_loss_per_symbol(symbol):

        try:

            connection = Database_Management.Create_My_Connection()
            mycursor = connection.cursor()
            #insert data int table
            mycursor.execute('SELECT Profit_Loss FROM {}_trading WHERE Good_Bad_Trade=\'BAD\';'.format(symbol))
            ihatePython = mycursor.fetchone()# in order to produce a "None" which describes an empty record you have to do this because pyodbc doesn't handle null records
            # #cursor.close()
            #print(ihatePython)

            if(ihatePython is None):
                return "None"

            else:
                mycursor.execute('SELECT Profit_Loss FROM {}_trading WHERE Good_Bad_Trade=\'BAD\';'.format(symbol))
                for i in mycursor:
                    i = str(i).replace("(","")
                    i = str(i).replace(",","")
                    i = str(i).replace(")","")

                    return float(i)
      

            mycursor.close()


        
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


    def Get_total_profit_gain_per_symbol_today(symbol):

        try:
            tday = datetime.datetime.today()
            todaysDate = tday.date()

            connection = Database_Management.Create_My_Connection()
            mycursor = connection.cursor()
            #insert data int table
            mycursor.execute('SELECT Profit_Loss FROM {}_trading WHERE Good_Bad_Trade=\'GOOD\' AND nDate=\'{}\';'.format(symbol,todaysDate))
            ihatePython = mycursor.fetchone()# in order to produce a "None" which describes an empty record you have to do this because pyodbc doesn't handle null records
        
            if(ihatePython is None):
                return "None"

            else:
                mycursor.execute('SELECT Profit_Loss FROM {}_trading WHERE Good_Bad_Trade=\'GOOD\' AND nDate=\'{}\';'.format(symbol,todaysDate))
                for i in mycursor:
                    i = str(i).replace("(","")
                    i = str(i).replace(",","")
                    i = str(i).replace(")","")

                    return float(i)
      

            mycursor.close()


        
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


    def Get_total_profit_loss_per_symbol_today(symbol):

        try:
            tday = datetime.datetime.today()
            todaysDate = tday.date()

            connection = Database_Management.Create_My_Connection()
            mycursor = connection.cursor()
            #insert data int table
            mycursor.execute('SELECT Profit_Loss FROM {}_trading WHERE Good_Bad_Trade=\'BAD\' AND nDate=\'{}\';'.format(symbol,todaysDate))
            ihatePython = mycursor.fetchone()# in order to produce a "None" which describes an empty record you have to do this because pyodbc doesn't handle null records
        
            if(ihatePython is None):
                return "None"

            else:
                mycursor.execute('SELECT Profit_Loss FROM {}_trading WHERE Good_Bad_Trade=\'BAD\' AND nDate=\'{}\';'.format(symbol,todaysDate))
                for i in mycursor:
                    i = str(i).replace("(","")
                    i = str(i).replace(",","")
                    i = str(i).replace(")","")

                    return float(i)
      

            mycursor.close()


        
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


    def Get_total_profit_for_all_symbols():

        profitArray = []
        

        try:

            symbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()
            counter = 1
            for symbol in symbolList:

                profit = Interface_Management.Get_total_profit_gain_per_symbol(symbol)
                loss = Interface_Management.Get_total_profit_loss_per_symbol(symbol)
                #print(profit)
                profitArray.append(profit)
                profitArray.append(loss)
                if(str(profit) == "None"):
                    profitArray.remove(profitArray[-2])
                if(str(loss) == "None"):
                    profitArray.remove(profitArray[-1])
        
                if counter > 170:
                    break                    
                    

                counter += 1


            totalSum = sum(profitArray)

            return totalSum

                   
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)

    
    def Get_total_profit_for_all_symbols_today():

        profitArray = []
        

        try:

            symbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()
            counter = 1
            for symbol in symbolList:

                profit = Interface_Management.Get_total_profit_gain_per_symbol_today(symbol)
                loss = Interface_Management.Get_total_profit_loss_per_symbol_today(symbol)
                #print(profit)
                profitArray.append(profit)
                profitArray.append(loss)
                if(str(profit) == "None"):
                    profitArray.remove(profitArray[-2])
                if(str(loss) == "None"):
                    profitArray.remove(profitArray[-1])
        
                if counter > 170:
                    break                    
                    

                counter += 1


            totalSum = sum(profitArray)

            return totalSum

                   
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


    def Get_total_ROI_gain_per_symbol(symbol):

        try:

            connection = Database_Management.Create_My_Connection()
            mycursor = connection.cursor()
            #insert data int table
            mycursor.execute('SELECT ROI FROM {}_trading WHERE Good_Bad_Trade=\'GOOD\';'.format(symbol))
            ihatePython = mycursor.fetchone()# in order to produce a "None" which describes an empty record you have to do this because pyodbc doesn't handle null records
            # #cursor.close()
            #print(ihatePython)

            if(ihatePython is None):
                return "None"

            else:
                mycursor.execute('SELECT ROI FROM {}_trading WHERE Good_Bad_Trade=\'GOOD\';'.format(symbol))
                for i in mycursor:
                    i = str(i).replace("(","")
                    i = str(i).replace(",","")
                    i = str(i).replace(")","")

                    return float(i)
      

            mycursor.close()


        
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)

    
    def Get_total_ROI_loss_per_symbol(symbol):

        try:

            connection = Database_Management.Create_My_Connection()
            mycursor = connection.cursor()
            #insert data int table
            mycursor.execute('SELECT ROI FROM {}_trading WHERE Good_Bad_Trade=\'BAD\';'.format(symbol))
            ihatePython = mycursor.fetchone()# in order to produce a "None" which describes an empty record you have to do this because pyodbc doesn't handle null records
            # #cursor.close()
            #print(ihatePython)

            if(ihatePython is None):
                return "None"

            else:
                mycursor.execute('SELECT ROI FROM {}_trading WHERE Good_Bad_Trade=\'BAD\';'.format(symbol))
                for i in mycursor:
                    i = str(i).replace("(","")
                    i = str(i).replace(",","")
                    i = str(i).replace(")","")

                    return float(i)
      

            mycursor.close()


        
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)

    
    def Get_AVG_ROI_for_all_symbols():

        profitArray = []
        

        try:

            symbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()
            counter = 1
            for symbol in symbolList:

                profit = Interface_Management.Get_total_ROI_gain_per_symbol(symbol)
                loss = Interface_Management.Get_total_ROI_loss_per_symbol(symbol)
                #print(profit)
                profitArray.append(profit)
                profitArray.append(loss)
                if(str(profit) == "None"):
                    profitArray.remove(profitArray[-2])
                if(str(loss) == "None"):
                    profitArray.remove(profitArray[-1])
        
                if counter > 170:
                    break                    
                    

                counter += 1


            totalSum = sum(profitArray)
            avg = totalSum/len(profitArray)

            return avg

                   
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


    def Get_AVG_GOOD_trades():

        profitArray = []
        goodArray = []
        

        try:

            symbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()
            counter = 1
            for symbol in symbolList:

                profit = Interface_Management.Get_total_ROI_gain_per_symbol(symbol)
                loss = Interface_Management.Get_total_ROI_loss_per_symbol(symbol)
                #print(profit)
                goodArray.append(profit)
                profitArray.append(profit)
                profitArray.append(loss)
                if(str(profit) == "None"):
                    goodArray.remove(profitArray[-2])
                    profitArray.remove(profitArray[-2])
                if(str(loss) == "None"):
                    profitArray.remove(profitArray[-1])
        
                if counter > 170:
                    break                    
                    

                counter += 1

            amountGoodTrades = len(goodArray)
            totalAmountTrades = len(profitArray)
            fractionForm = str(amountGoodTrades)+"/"+str(totalAmountTrades)
            good_vs_total = fractionForm, amountGoodTrades/totalAmountTrades

            return good_vs_total

                   
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


class Logging:       

    def error(message):

        try:


            tday = datetime.datetime.today()
            todaysDate = tday.date()
            now = datetime.datetime.now()
            todaysName = Date_Time_Management.Get_Todays_Weekday_Name()


            filename="Logs\\{}.log".format(todaysDate)

            f = open(filename, "a")
            f.write(corama.Fore.BLUE+"*******************\f")
            f.write(corama.Fore.BLUE+"*******************\r\f")
            f.write("{} - {}".format(todaysName, now)+" - ERROR\f")
            f.write("{}\r\f".format(message))

            f.close()

        except:
            error = traceback.format_exc()
            Logging.error(error)


    def error_Trading(message):

        tday = datetime.datetime.today()
        todaysDate = tday.date()
        now = datetime.datetime.now()
        todaysName = Date_Time_Management.Get_Todays_Weekday_Name()

        filename="Logs\\{}_Trading.log".format(todaysDate)

        f = open(filename, "a")
        f.write(corama.Fore.BLUE+"*******************\f")
        f.write(corama.Fore.BLUE+"*******************\r\f")
        f.write("{} - {}".format(todaysName, now)+" - ERROR\f")
        f.write("{}\r\f".format(message))

        f.close()

    def info(message):

        try:

            tday = datetime.datetime.today()
            todaysDate = tday.date()
            now = datetime.datetime.now()
            todaysName = Date_Time_Management.Get_Todays_Weekday_Name()

            filename="Logs\\{}.log".format(todaysDate)

            f = open(filename, "a")
            f.write(corama.Fore.BLUE+"-------------------\f")
            f.write(corama.Fore.BLUE+"-------------------\r\f")
            f.write("{} - {}".format(todaysName, now)+" - INFO\f")
            f.write("{}\r\f".format(message))

            f.close()
        except:
            error = traceback.format_exc()
            Logging.error(error)

    def info_Trading(message):

        tday = datetime.datetime.today()
        todaysDate = tday.date()
        now = datetime.datetime.now()

        todaysName = Date_Time_Management.Get_Todays_Weekday_Name()

        filename="Logs\\{}_Trading.log".format(todaysDate)

        f = open(filename, "a")
        f.write(corama.Fore.BLUE+"-------------------\f")
        f.write(corama.Fore.BLUE+"-------------------\r\f")
        f.write("{} - {}".format(todaysName, now)+" - INFO\f")
        f.write("{}\r\f".format(message))

        f.close()


    def test(fileLabel, message):
    
        tday = datetime.datetime.today()
        todaysDate = tday.date()        

        filename="Testing\\{}_{}.test".format(todaysDate, fileLabel)

        f = open(filename, "a")        
        f.write("{}".format(message)+"\f")

        f.close()


    def test_Trading(fileLabel, message):
    
        tday = datetime.datetime.today()
        todaysDate = tday.date()        

        filename="Testing\\{}_{}_Trading.test".format(todaysDate, fileLabel)

        f = open(filename, "a")        
        f.write("{}".format(message)+"\f")

        f.close()


class Testing:

    def Test_See_if_goodBuys_Are_Occurring_in_DB(howManyRecordsToCheck):

        try:
            print("Pulling symbol list from LLDB...")

            symbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()

            for symbol in symbolList:

                

                tableName = "{}_historical".format(symbol)

                if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)):
                    

                    tday = datetime.datetime.today()
                    todaysDate = str(tday.date())
                    now = tday.time()

                    array = []

                    difference = howManyRecordsToCheck

                    newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Last_Row_of_Historical_Tables(symbol)

                    dBdate = str(newestRow[1]).replace(" '","")
                    dBdate = dBdate.replace("'","")
                    dBdate = dBdate.replace(" ","")
                    time = str(newestRow[2]).replace(" '","")
                    time = time.replace("'","")
                    time = time.replace(" ","")

                    time = datetime.datetime.strptime(time, '%H:%M:%S')

                    

                    newestIndex = int(newestRow[31])

                    index = newestIndex - difference

                    tday = datetime.datetime.today()
                    todaysDate = str(tday.date())
                    now = tday.time()
                    endTime = now.replace(hour=15, minute=59, second=0)
                    startTime = now.replace(hour=9, minute=29, second=0)


                    while(index <= newestIndex):

                        newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index)
                        previousRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index-1)

                        if(newestRow == "" or previousRow == ""):
                            
                            newest_ema200_value = 0
                            previous_ema200_value = 0
                            
                            closePrice = 0
                            previousClosePrice = 0

                            dBdate = todaysDate
                            time = now

                            newestAngle = 0
                            previousAngle = 0
                    

                            avgStdDev10 = 0
                            StdDev10 = 0
                            hlev_percentage = 0
                            ema200_value = 0
                            hlev2000_UpperArm = 0
                            hlevOrigin = 0

                    
                            rowID = 1

                        else:

                            #print("in here")
                            newestAngle = float(newestRow[8])
                            previousAngle = float(previousRow[8])
                            newest_NegStdDev2 = float(newestRow[20])
                            previous_NegStdDev2 = float(newestRow[20])
                            newest_NegStdDev3 = float(newestRow[21])
                            previous_NegStdDev3 = float(previousRow[21])

                            avgStdDev10 = float(newestRow[29])
                            StdDev10 = float(newestRow[18])
                            hlev_percentage = float(newestRow[32])
                            ema200_value = float(newestRow[7])
                            hlev2000_UpperArm = float(newestRow[33])
                            hlev2000_LowerArm = float(newestRow[35])
                            hlevOrigin = float(newestRow[30])

                            
                            newest_ema200_value = float(newestRow[7])
                            previous_ema200_value = float(previousRow[7])
                            
                            #filter out lower value stocks that could hit the danger zone for 
                            closePrice = float(newestRow[3])
                            if(closePrice <=5.00):
                                return
                            previousClosePrice = float(previousRow[3])

                            dBdate = str(newestRow[1]).replace(" '","")
                            dBdate = dBdate.replace("'","")
                            dBdate = dBdate.replace(" ","")

                            time = str(newestRow[2]).replace(" '","")
                            time = time.replace("'","")
                            time = time.replace(" ","")

                            time = datetime.datetime.strptime(time, '%H:%M:%S')


                            index = index + 1


                            print("{} Checking symbol {}...".format(index, symbol))
                            
                            with(open("C:\\Users\\"+windowsUsername+"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Testing\\GoodBuys.test", "a")) as f:
                                f.write("{} Checking symbol {}...\r".format(index, symbol))

                                            
                                    
                            if((previousAngle < previous_NegStdDev3 and newestAngle > newest_NegStdDev3) or (previousAngle < previous_NegStdDev2 and newestAngle > newest_NegStdDev2)): #emaAngle crosses above stdDev3 on todays date
                                #and avgStdDev10 < stdDev10
                                if(avgStdDev10 < StdDev10):

                                    if(hlev_percentage <= -1):

                                        if(hlev2000_UpperArm < hlevOrigin):

                                            if(ema200_value < hlev2000_LowerArm):  

                                                print("Good Buy detected for {} at index {}".format(symbol, index-1))  
                                                with(open("C:\\Users\\"+windowsUsername+"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Testing\\GoodBuys.test", "a")) as f:
                                                    f.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Good Buy detected for {} at index {}\r".format(symbol, index))                       
                            

        
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


    def Test_See_if_goodBuys_Are_Occurring_for_one_symbol(howManyRecordsToCheck, symbol):

        try:                       

            tableName = "{}_historical".format(symbol)

            if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)):
                

                tday = datetime.datetime.today()
                todaysDate = str(tday.date())
                now = tday.time()

                array = []

                difference = howManyRecordsToCheck

                newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Last_Row_of_Historical_Tables(symbol)

                dBdate = str(newestRow[1]).replace(" '","")
                dBdate = dBdate.replace("'","")
                dBdate = dBdate.replace(" ","")
                time = str(newestRow[2]).replace(" '","")
                time = time.replace("'","")
                time = time.replace(" ","")

                time = datetime.datetime.strptime(time, '%H:%M:%S')

                

                newestIndex = int(newestRow[31])

                index = newestIndex - difference

                tday = datetime.datetime.today()
                todaysDate = str(tday.date())
                now = tday.time()
                endTime = now.replace(hour=15, minute=59, second=0)
                startTime = now.replace(hour=9, minute=29, second=0)


                while(index <= newestIndex):

                    newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index)
                    previousRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index-1)

                    if(newestRow == "" or previousRow == ""):
                        
                        newest_ema200_value = 0
                        previous_ema200_value = 0
                        
                        closePrice = 0
                        previousClosePrice = 0

                        dBdate = todaysDate
                        time = now

                        newestAngle = 0
                        previousAngle = 0
                

                        avgStdDev10 = 0
                        StdDev10 = 0
                        hlev_percentage = 0
                        ema200_value = 0
                        hlev2000_UpperArm = 0
                        hlevOrigin = 0

                
                        rowID = 1

                    else:

                        #print("in here")
                        newestAngle = float(newestRow[8])
                        previousAngle = float(previousRow[8])
                        newest_NegStdDev2 = float(newestRow[20])
                        previous_NegStdDev2 = float(newestRow[20])
                        newest_NegStdDev3 = float(newestRow[21])
                        previous_NegStdDev3 = float(previousRow[21])

                        avgStdDev10 = float(newestRow[29])
                        StdDev10 = float(newestRow[18])
                        hlev_percentage = float(newestRow[32])
                        ema200_value = float(newestRow[7])
                        hlev2000_UpperArm = float(newestRow[33])
                        hlev2000_LowerArm = float(newestRow[35])
                        hlevOrigin = float(newestRow[30])

                        
                        newest_ema200_value = float(newestRow[7])
                        previous_ema200_value = float(previousRow[7])
                        
                        #filter out lower value stocks that could hit the danger zone for 
                        closePrice = float(newestRow[3])
                        if(closePrice <=5.00):
                            return
                        previousClosePrice = float(previousRow[3])

                        dBdate = str(newestRow[1]).replace(" '","")
                        dBdate = dBdate.replace("'","")
                        dBdate = dBdate.replace(" ","")

                        time = str(newestRow[2]).replace(" '","")
                        time = time.replace("'","")
                        time = time.replace(" ","")

                        time = datetime.datetime.strptime(time, '%H:%M:%S')


                        index = index + 1


                        print("{} Checking symbol {}...".format(index, symbol))
                        
                        with(open("C:\\Users\\"+windowsUsername+"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Testing\\GoodBuys.test", "a")) as f:
                            f.write("{} Checking symbol {}...\r".format(index, symbol))

                                        
                                
                        if((previousAngle < previous_NegStdDev3 and newestAngle > newest_NegStdDev3) or (previousAngle < previous_NegStdDev2 and newestAngle > newest_NegStdDev2)): #emaAngle crosses above stdDev3 on todays date
                            #and avgStdDev10 < stdDev10
                            if(avgStdDev10 < StdDev10):

                                if(hlev_percentage <= -1):

                                    if(hlev2000_UpperArm < hlevOrigin):

                                        if(ema200_value < hlev2000_LowerArm):  

                                            print("Good Buy detected for {} at index {}".format(symbol, index-1))  
                                            with(open("C:\\Users\\"+windowsUsername+"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Testing\\GoodBuys.test", "a")) as f:
                                                f.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Good Buy detected for {} at index {}\r".format(symbol, index))                       
                        

        
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


    def Test_See_if_goodSells_Are_Occurring_in_DB(howManyRecordsToCheck):

        try:
            print("Pulling symbol list from LLDB...")

            symbolList = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Symbol_List()

            for symbol in symbolList:

                

                tableName = "{}_historical".format(symbol)

                if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)):
                    

                    tday = datetime.datetime.today()
                    todaysDate = str(tday.date())
                    now = tday.time()

                    array = []

                    difference = howManyRecordsToCheck

                    newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Last_Row_of_Historical_Tables(symbol)

                    dBdate = str(newestRow[1]).replace(" '","")
                    dBdate = dBdate.replace("'","")
                    dBdate = dBdate.replace(" ","")
                    time = str(newestRow[2]).replace(" '","")
                    time = time.replace("'","")
                    time = time.replace(" ","")

                    time = datetime.datetime.strptime(time, '%H:%M:%S')

                    

                    newestIndex = int(newestRow[31])

                    index = newestIndex - difference

                    tday = datetime.datetime.today()
                    todaysDate = str(tday.date())
                    now = tday.time()
                    endTime = now.replace(hour=15, minute=59, second=0)
                    startTime = now.replace(hour=9, minute=29, second=0)


                    while(index <= newestIndex):

                        newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index)
                        previousRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index-1)

                        if(newestRow == "" or previousRow == ""):
                            
                            newest_ema200_value = 0
                            previous_ema200_value = 0
                            
                            closePrice = 0
                            previousClosePrice = 0

                            dBdate = todaysDate
                            time = now

                            newestAngle = 0
                            previousAngle = 0
                    

                            avgStdDev10 = 0
                            StdDev10 = 0
                            hlev_percentage = 0
                            ema200_value = 0
                            hlev2000_UpperArm = 0
                            hlevOrigin = 0

                    
                            rowID = 1

                        else:

                            #print("in here")
                            newestAngle = float(newestRow[8])
                            previousAngle = float(previousRow[8])
                            newest_NegStdDev2 = float(newestRow[20])
                            previous_NegStdDev2 = float(newestRow[20])
                            newest_NegStdDev3 = float(newestRow[21])
                            previous_NegStdDev3 = float(previousRow[21])

                            avgStdDev10 = float(newestRow[29])
                            StdDev10 = float(newestRow[18])
                            hlev_percentage = float(newestRow[32])
                            ema200_value = float(newestRow[7])
                            hlev2000_UpperArm = float(newestRow[33])
                            hlev2000_LowerArm = float(newestRow[35])
                            hlevOrigin = float(newestRow[30])

                            
                            newest_ema200_value = float(newestRow[7])
                            previous_ema200_value = float(previousRow[7])
                            
                            #filter out lower value stocks that could hit the danger zone for 
                            closePrice = float(newestRow[3])
                            if(closePrice <=5.00):
                                return
                            previousClosePrice = float(previousRow[3])

                            dBdate = str(newestRow[1]).replace(" '","")
                            dBdate = dBdate.replace("'","")
                            dBdate = dBdate.replace(" ","")

                            time = str(newestRow[2]).replace(" '","")
                            time = time.replace("'","")
                            time = time.replace(" ","")

                            time = datetime.datetime.strptime(time, '%H:%M:%S')


                            index = index + 1


                            print("{} Checking symbol {}...".format(index, symbol))
                            
                            with(open("C:\\Users\\"+windowsUsername+"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Testing\\GoodBuys.test", "a")) as f:
                                f.write("{} Checking symbol {}...\r".format(index, symbol))

                                            
                                    
                            if((previousAngle < previous_NegStdDev3 and newestAngle > newest_NegStdDev3) or (previousAngle < previous_NegStdDev2 and newestAngle > newest_NegStdDev2)): #emaAngle crosses above stdDev3 on todays date
                                #and avgStdDev10 < stdDev10
                                if(avgStdDev10 < StdDev10):

                                    if(hlev_percentage <= -1):

                                        if(hlev2000_UpperArm < hlevOrigin):

                                            if(ema200_value < hlev2000_LowerArm):  

                                                print("Good Buy detected for {} at index {}".format(symbol, index-1))  
                                                with(open("C:\\Users\\"+windowsUsername+"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Testing\\GoodBuys.test", "a")) as f:
                                                    f.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Good Buy detected for {} at index {}\r".format(symbol, index))                       
                            

        
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)

    
    def Test_Historical_Day_Data_from_Schwab():

        symbol = "AAPL"
        startDateTime = dt(2025, 1, 1)
        endDateTime = dt(2025, 6, 1)

        try:

            authorization = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
            priceHistory = authorization.get_price_history_every_day(symbol, start_datetime=startDateTime, end_datetime=endDateTime, need_extended_hours_data="yes")

            #print(priceHistory)

            Logging.test("Test_Historical_Day_Data_from_Schwab", json.dumps(priceHistory.json(), indent=4))

            return 

        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)


    def Test_if_TradeData_From_Schwab_has_changed():

        #this is trade data from stock AES. A buy and sell from the same trade.
        #previous trade data checked on 2025-09-14 and there is test log file for that date.

        orderID = "1004102590581"
        tradeData = Pull_From_Schwab_Management.Get_Trade_With_Order_ID(orderID)
                # stuff = json.loads(json.dumps(tradeData.json(), indent=4))
        stuff = json.dumps(tradeData, indent=4)

        Logging.test("TradeData_BuyDataTest", str(stuff))


        orderID = "1004132412296"
        tradeData = Pull_From_Schwab_Management.Get_Trade_With_Order_ID(orderID)
        stuff = json.dumps(tradeData, indent=4)

        Logging.test("TradeData_SellDataTest", str(stuff))


class Trading_Management:

    def LootLoader_Method(symbol):


        todaysDay = Date_Time_Management.Get_Todays_Weekday_Name()

        if(todaysDay == "Saturday" and todaysDay == "Sunday"):
            #exit method
            print("Stock market closed on saturday and sunday. No buying or selling allowed")

            return

        # This method will do all of the checking that lootloader does
        # This will be implemented at the end of dataloaders process instead 
        # of being its own dedicated program, separate from dataloader. This
        # enables the program not to have to worry about separate timing

        #first check the trading status

        tradeStatus = Trading_Management.Get_Trade_Status(symbol)
        message = "{} Trade Status: ".format(symbol)+str(tradeStatus)
        print(message)
        Logging.info(message)
        

        # if(str(tradeStatus) == "(0, 0, 0, 0)"):#if(lookingForBuy == 0 and okToBuy == 0 and lookingForSell == 0 and okToSell == 0):# looking for buy entry
        #     #updatedStatus = Trading_Management.Looking_For_A_Buy_Method(symbol)
        #     updatedStatus = Trading_Management.Buy_without_LookingForBuy_Method(symbol)
            # updatedStatus = False

            # tradeStatus = Trading_Management.Get_Trade_Status(symbol)
            

            # if("True" in str(updatedStatus) and str(tradeStatus) == "(1, 0, 0, 0)"):
            #     print("{} Trade Status: ".format(symbol)+str(tradeStatus))
            #     updatedStatus = Trading_Management.Attempt_Buy_at_MarketValue(symbol)
            
        if(str(tradeStatus) == "(0, 0, 0, 0)"):#if(lookingForBuy == 1 and okToBuy == 0 and lookingForSell == 0 and okToSell == 0):# looking for buy entry
            updatedStatus = Trading_Management.Buy_without_LookingForBuy_Method(symbol)
        # if(str(tradeStatus) == "(1, 1, 0, 0)"):#if(lookingForBuy == 1 and okToBuy == 1 and lookingForSell == 0 and okToSell == 0):# looking for buy entry
        #     pass
        if(str(tradeStatus) == "(1, 1, 1, 0)"):#if(lookingForBuy == 1 and okToBuy == 1 and lookingForSell == 1 and okToSell == 0):# looking for buy entry
            updatedStatus = Trading_Management.Attempt_Sell_at_MarketValue(symbol)
    
    def Get_Trade_Status(symbol):

        status = []

        lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
        #print(str(lastTradingRow))
        if(str(lastTradingRow) == "None"):
            #set trade status to neutral - not looking, ok to buy, not ok to sell
            lookingForBuy = 0
            lookingForSell = 0
            okToBuy = 0
            okToSell = 0

            status = lookingForBuy, okToBuy, lookingForSell, okToSell

            return status


        if(str(lastTradingRow) != "None"):

            #trade status
            if(str(lastTradingRow[4]) == "0"):
                lookingForBuy = 0
            if(str(lastTradingRow[4]) == "1"):
                lookingForBuy = 1 
            if(str(lastTradingRow[5] )== "0"):
                okToBuy = 0
            if(str(lastTradingRow[5]) == "1"):
                okToBuy = 1
            if(str(lastTradingRow[6])== "0"):
                lookingForSell = 0
            if(str(lastTradingRow[6]) == "1"):
                lookingForSell = 1
            if(str(lastTradingRow[7])== "0"):
                okToSell = 0
            if(str(lastTradingRow[7]) == "1"):
                okToSell = 1

            status = lookingForBuy, okToBuy, lookingForSell, okToSell

            return status
    
    def Looking_For_A_Buy_Method(symbol):

        try:
            tday = datetime.datetime.today()
            todaysDate = str(tday.date())
            now = tday.time()

            array = []

            difference = 45

            newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Last_Row_of_Historical_Tables(symbol)
            
            if(newestRow is None):
                return
            #print(newestRow)


            # dBdate = str(newestRow[1]).replace(" '","")
            # dBdate = dBdate.replace("'","")
            # dBdate = dBdate.replace(" ","")

            # time = str(newestRow[2]).replace(" '","")
            # time = time.replace("'","")
            # time = time.replace(" ","")
            

            # time = datetime.datetime.strptime(time, '%H:%M:%S')

            # hour = now.replace(hour=1, minute=00, second=0)
            # thirtyMins = now.replace(hour=00, minute=30, second=0)
            # fifteenMins = now.replace(hour=00, minute=15, second=0)

            # # halfHourAgo = datetime.datetime.combine(date.min, now) - datetime.datetime.combine(date.min,thirtyMins)
            # # halfHourAgo = str(halfHourAgo)
            # # halfHourAgo = datetime.datetime.strptime(halfHourAgo, '%H:%M:%S')

            # # fifteenMinsAgo = datetime.datetime.combine(date.min, now) - datetime.datetime.combine(date.min,fifteenMins)
            # # fifteenMinsAgo = str(fifteenMinsAgo)
            # # fifteenMinsAgo = datetime.datetime.strptime(fifteenMinsAgo, '%H:%M:%S')

            # hourAgo = datetime.datetime.combine(date.min, now) - datetime.datetime.combine(date.min,hour)
            # hourAgo = str(hourAgo)
            # hourAgo = datetime.datetime.strptime(hourAgo, '%H:%M:%S')


            newestIndex = int(newestRow[31])

            index = newestIndex - difference

            
            while(index <= newestIndex):
                #print(index)
                newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index)
                previousRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index-1)
                #print(newestRow)
                if(newestRow == "" or previousRow == ""):

                    newestAngle = 0
                    previousAngle = 0
               

                    avgStdDev10 = 0
                    StdDev10 = 0
                    hlev_percentage = 0
                    ema200_value = 0
                    hlev2000_UpperArm = 0
                    hlevOrigin = 0

                    dBdate = todaysDate
                    time = now
                    rowID = 1
                else:
                    #print("in here")
                    newestAngle = float(newestRow[8])
                    previousAngle = float(previousRow[8])
                    newest_NegStdDev2 = float(newestRow[20])
                    previous_NegStdDev2 = float(newestRow[20])
                    newest_NegStdDev3 = float(newestRow[21])
                    previous_NegStdDev3 = float(previousRow[21])

                    avgStdDev10 = float(newestRow[29])
                    StdDev10 = float(newestRow[18])
                    hlev_percentage = float(newestRow[32])
                    ema200_value = float(newestRow[7])
                    hlev2000_UpperArm = float(newestRow[33])
                    hlev2000_LowerArm = float(newestRow[35])
                    hlevOrigin = float(newestRow[30])



                    dBdate = str(newestRow[1]).replace(" '","")
                    dBdate = dBdate.replace("'","")
                    dBdate = dBdate.replace(" ","")

                    time = str(newestRow[2]).replace(" '","")
                    time = time.replace("'","")
                    time = time.replace(" ","")

                    time = datetime.datetime.strptime(time, '%H:%M:%S')


                    index = index + 1

                    #ema200Angle crosses above NegstdDev3 or ema200Angle crosses above NegstdDev2
                    if((previousAngle < previous_NegStdDev3 and newestAngle > newest_NegStdDev3) or (previousAngle < previous_NegStdDev2 and newestAngle > newest_NegStdDev2)): #emaAngle crosses above stdDev3 on todays date
                        #and avgStdDev10 < stdDev10
                        if(avgStdDev10 < StdDev10):

                            if(hlev_percentage <= -1):

                                if(hlev2000_UpperArm < hlevOrigin):

                                    if(ema200_value < hlev2000_LowerArm):
                                        
                                        time = str(time).replace("1900-01-01 ","")
                                        #print("True "+str(dBdate)+" "+str(time))
                                        localDate = dBdate
                                        localTime = time                        
                                        lookingForBuy = 1
                                        alreadyBought = 0
                                        lookingForSell = 0
                                        alreadySold = 0
                                        orderType = "LKFB"                                                              
                                        Profit_or_Loss = 0.00
                                        roi = 0.00
                                        Good_Bad_Trade = "NA"
                                        orderID = 0
                                        riskAmount = 0
                                        accountBalance = 0
                                        closePrice = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Latest_ClosePrice()
                                        numberOfShares = 0
                                        buyPrice = 0
                                        lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                                        if(lastTradingRow == "None"):
                                            rowID = 1
                                        if(lastTradingRow != "None"):
                                            rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                                        incomeTaxesDue = 0
                                        sellPrice = 0
                                        Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                                lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                                alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                                riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                                orderID, rowID)
                                        #-------------

                                        fileLabel = "{}_lookForBuyTest".format(symbol)
                                        
                                        
                                        time = str(time).replace("1900-01-01 ","")
                                        message = symbol, True, dBdate, time
                                        print(message)
                                        Logging.info(message)
                                        array.append(True)              
                                        array.append(dBdate)                                                     
                                        array.append(time) 
                                        return array

                
            else:

                array.append(False)              
                array.append(dBdate) 
                fileLabel = "{}_lookForBuyTest".format(symbol)
                                
                                
                time = str(time).replace("1900-01-01 ","")
                message = symbol, False, dBdate, time
                print(message)
                Logging.info(message)             
                array.append(str(time)) 
                return array
        except:
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)

        
    def Buy_without_LookingForBuy_Method(symbol):

        try:

            tableName = "{}_historical".format(symbol)

            if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)):


                tday = datetime.datetime.today()
                todaysDate = str(tday.date())
                now = tday.time()

                array = []

                difference = 45

                newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Last_Row_of_Historical_Tables(symbol)
                if(newestRow is None):
                    return
                dBdate = str(newestRow[1]).replace(" '","")
                dBdate = dBdate.replace("'","")
                dBdate = dBdate.replace(" ","")
                time = str(newestRow[2]).replace(" '","")
                time = time.replace("'","")
                time = time.replace(" ","")

                time = datetime.datetime.strptime(time, '%H:%M:%S')

                

                newestIndex = int(newestRow[31])

                index = newestIndex - difference

                tday = datetime.datetime.today()
                todaysDate = str(tday.date())
                now = tday.time()
                endTime = now.replace(hour=15, minute=59, second=0)
                startTime = now.replace(hour=9, minute=29, second=0)


                while(index <= newestIndex):

                    newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index)
                    previousRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index-1)

                    if(newestRow == "" or previousRow == ""):
                        
                        newest_ema200_value = 0
                        previous_ema200_value = 0
                        
                        closePrice = 0
                        previousClosePrice = 0

                        dBdate = todaysDate
                        time = now

                        newestAngle = 0
                        previousAngle = 0
                

                        avgStdDev10 = 0
                        StdDev10 = 0
                        hlev_percentage = 0
                        ema200_value = 0
                        hlev2000_UpperArm = 0
                        hlevOrigin = 0

                 
                        rowID = 1

                    else:

                        #print("in here")
                        newestAngle = float(newestRow[8])
                        previousAngle = float(previousRow[8])
                        newest_NegStdDev2 = float(newestRow[20])
                        previous_NegStdDev2 = float(newestRow[20])
                        newest_NegStdDev3 = float(newestRow[21])
                        previous_NegStdDev3 = float(previousRow[21])

                        avgStdDev10 = float(newestRow[29])
                        StdDev10 = float(newestRow[18])
                        hlev_percentage = float(newestRow[32])
                        ema200_value = float(newestRow[7])
                        hlev2000_UpperArm = float(newestRow[33])
                        hlev2000_LowerArm = float(newestRow[35])
                        hlevOrigin = float(newestRow[30])

                        
                        newest_ema200_value = float(newestRow[7])
                        previous_ema200_value = float(previousRow[7])
                        
                        #filter out lower value stocks that could hit the danger zone for 
                        closePrice = float(newestRow[3])
                        if(closePrice <=5.00):
                            return
                        previousClosePrice = float(previousRow[3])

                        dBdate = str(newestRow[1]).replace(" '","")
                        dBdate = dBdate.replace("'","")
                        dBdate = dBdate.replace(" ","")

                        time = str(newestRow[2]).replace(" '","")
                        time = time.replace("'","")
                        time = time.replace(" ","")

                        time = datetime.datetime.strptime(time, '%H:%M:%S')


                        index = index + 1


                    

                        accountBalance = Account_Management.Get_Account_Balance_Available_For_Trading()
                        #put accountBalance into the LLDB
                        numberOfShares = Calculation_Management.Calculate_Share_Amount(closePrice, accountBalance)
                        riskAmount = closePrice * numberOfShares
                        tradeStatus = Trading_Management.Get_Trade_Status(symbol)
                      
                        #start the trade status over if we can't enter the trade because we don't have enough buying power
                        if((accountBalance - riskAmount) <= 1.00):

                            if(tradeStatus  == "(1, 0, 0, 0)"):

                                Database_Management.Delete_Data_From_LootLoaderDataBase.Trading.Delete_the_Last_Trading_Row_For_Symbol(symbol)
                                return
                        if(((accountBalance - riskAmount) > 100.00) and (startTime < now and now < endTime)):
                                          
                                
                            if((previousAngle < previous_NegStdDev3 and newestAngle > newest_NegStdDev3) or (previousAngle < previous_NegStdDev2 and newestAngle > newest_NegStdDev2)): #emaAngle crosses above stdDev3 on todays date
                                #and avgStdDev10 < stdDev10
                                if(avgStdDev10 < StdDev10):

                                    if(hlev_percentage <= -1):

                                        if(hlev2000_UpperArm < hlevOrigin):

                                            if(ema200_value < hlev2000_LowerArm):                           
                            

                                                #send buy signal to charles schwab (still need to put STOP value in here)
                                                #------------------------------------------------------------------------------------------------            
                                                accountHash = Account_Management.Get_Account_Hash()
                                                auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
                                                order = schwab.orders.equities.equity_buy_market(symbol, numberOfShares).set_duration(Duration.GOOD_TILL_CANCEL)
                                                r = auth.place_order(accountHash, order)
                                                tradeID = schwab.utils.Utils(auth, accountHash).extract_order_id(r)
                                                #------------------------------------------------------------------------------------------------                       


                                                #put the trades data into the trades database
                                                #------------------------------------------------------------------------------------------------                        
                                                localDate = dBdate
                                                time = str(time).replace("1900-01-01 ","") 
                                                localTime = time                        
                                                lookingForBuy = 1
                                                alreadyBought = 1
                                                lookingForSell = 1
                                                alreadySold = 0
                                                orderType = "BUY"                                                              
                                                Profit_or_Loss = 0.00
                                                roi = 0.00
                                                Good_Bad_Trade = "NA"
                                                orderID = tradeID
                                                                        
                                                lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                                                if(str(lastTradingRow) == "None"):
                                                    rowID = 1
                                                else:
                                                    rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                                                buyPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID)
                                                riskAmount = buyPrice * numberOfShares 
                                                incomeTaxesDue = 0
                                                sellPrice = 0
                                                Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                                        lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                                        alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                                        riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                                        orderID, rowID)
                                                #------------------------------------------------------------------------------------------------                

                                                time = str(time).replace("1900-01-01 ","")
                                                message = symbol+" True "+str(dBdate)+" "+str(time)+"\r***************************************Symbol {} bought {} shares at {}.**************************************".format(symbol, numberOfShares, buyPrice)
                                                print(message)
                                                Logging.info(message)

                                                # message = "***************************************Symbol {} bought {} shares at {}.**************************************".format(symbol, numberOfShares, buyPrice)
                                                # print(message)
                                                # Logging.info(message)
                                                return
                else:
                    time = str(time).replace("1900-01-01 ","") 
                    message = symbol+" False: {} {}".format(dBdate, time)
                    print(message)
                    Logging.info(message)
                    array.append(False)              
                    array.append(dBdate)                                 
                    array.append(time) 
                    return array




        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)
         

    def Attempt_Buy_at_MarketValue(symbol):

        try:

            tableName = "{}_historical".format(symbol)

            if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)):


                tday = datetime.datetime.today()
                todaysDate = str(tday.date())
                now = tday.time()

                array = []

                difference = 45

                newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Last_Row_of_Historical_Tables(symbol)
                if(newestRow is None):
                    return
                dBdate = str(newestRow[1]).replace(" '","")
                dBdate = dBdate.replace("'","")
                dBdate = dBdate.replace(" ","")
                time = str(newestRow[2]).replace(" '","")
                time = time.replace("'","")
                time = time.replace(" ","")

                time = datetime.datetime.strptime(time, '%H:%M:%S')

                

                newestIndex = int(newestRow[31])

                index = newestIndex - difference

                tday = datetime.datetime.today()
                todaysDate = str(tday.date())
                now = tday.time()
                endTime = now.replace(hour=14, minute=59, second=0)
                startTime = now.replace(hour=8, minute=29, second=0)


                while(index <= newestIndex):

                    newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index)
                    previousRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index-1)

                    if(newestRow == "" or previousRow == ""):
                        
                        newest_ema200_value = 0
                        previous_ema200_value = 0
                        
                        closePrice = 0
                        previousClosePrice = 0

                        dBdate = todaysDate
                        time = now

                    else:
                        
                        newest_ema200_value = float(newestRow[7])
                        previous_ema200_value = float(previousRow[7])
                        
                        closePrice = float(newestRow[3])
                        previousClosePrice = float(previousRow[3])

                        dBdate = str(newestRow[1]).replace(" '","")
                        dBdate = dBdate.replace("'","")
                        dBdate = dBdate.replace(" ","")

                        time = str(newestRow[2]).replace(" '","")
                        time = time.replace("'","")
                        time = time.replace(" ","")

                        time = datetime.datetime.strptime(time, '%H:%M:%S')


                        index = index + 1


                    

                        accountBalance = Account_Management.Get_Account_Balance_Available_For_Trading()
                        #put accountBalance into the LLDB
                        numberOfShares = Calculation_Management.Calculate_Share_Amount(closePrice, accountBalance)
                        riskAmount = closePrice * numberOfShares
                        tradeStatus = Trading_Management.Get_Trade_Status(symbol)
                      
                        #start the trade status over if we can't enter the trade
                        if((accountBalance - riskAmount) <= 1.00):

                            if(tradeStatus  == "(1, 0, 0, 0)"):

                                Database_Management.Delete_Data_From_LootLoaderDataBase.Trading.Delete_the_Last_Trading_Row_For_Symbol(symbol)
                                return
                        if(((accountBalance - riskAmount) > 100.00) and (startTime < now and now < endTime)):
                                          
                                
                            #look for buy entry

                            if( previousClosePrice < previous_ema200_value and closePrice > newest_ema200_value):                           
                            

                                #send buy signal to charles schwab (still need to put STOP value in here)
                                #------------------------------------------------------------------------------------------------            
                                accountHash = Account_Management.Get_Account_Hash()
                                auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
                                order = schwab.orders.equities.equity_buy_market(symbol, numberOfShares).set_duration(Duration.GOOD_TILL_CANCEL)
                                r = auth.place_order(accountHash, order)
                                tradeID = schwab.utils.Utils(auth, accountHash).extract_order_id(r)
                                #------------------------------------------------------------------------------------------------                       


                                #put the trades data into the trades database
                                #------------------------------------------------------------------------------------------------                        
                                localDate = dBdate
                                time = str(time).replace("1900-01-01 ","") 
                                localTime = time                        
                                lookingForBuy = 1
                                alreadyBought = 1
                                lookingForSell = 1
                                alreadySold = 0
                                orderType = "BUY"                                                              
                                Profit_or_Loss = 0.00
                                roi = 0.00
                                Good_Bad_Trade = "NA"
                                orderID = tradeID
                                                        
                                lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                                if(str(lastTradingRow) == "None"):
                                    rowID = 1
                                else:
                                    rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                                buyPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID)
                                riskAmount = buyPrice * numberOfShares 
                                incomeTaxesDue = 0
                                sellPrice = 0
                                Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                        lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                        alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                        riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                        orderID, rowID)
                                #------------------------------------------------------------------------------------------------                

                                time = str(time).replace("1900-01-01 ","")
                                message = symbol+" True "+str(dBdate)+" "+str(time)+"\r***************************************Symbol {} bought {} shares at {}.**************************************"
                                print(message)
                                Logging.info(message)

                                # message = "***************************************Symbol {} bought {} shares at {}.**************************************".format(symbol, numberOfShares, buyPrice)
                                # print(message)
                                # Logging.info(message)
                                return
                else:
                    time = str(time).replace("1900-01-01 ","") 
                    message = symbol+" False: {} {}".format(dBdate, time)
                    print(message)
                    Logging.info(message)
                    array.append(False)              
                    array.append(dBdate)                                 
                    array.append(time) 
                    return array




        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)

    def Attempt_Sell_at_MarketValue(symbol):

        try:

            tableName = "{}_historical".format(symbol)

            if(Database_Management.Check_LootLoaderDataBase.Check_If_Table_Exists_in_LLDB("lootloaderdatabase",tableName)):


                tday = datetime.datetime.today()
                todaysDate = str(tday.date())
                now = tday.time()

                array = []

                difference = 45

                newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Last_Row_of_Historical_Tables(symbol)
                if(newestRow is None):
                    return
                dBdate = str(newestRow[1]).replace(" '","")
                dBdate = dBdate.replace("'","")
                dBdate = dBdate.replace(" ","")

                newestIndex = int(newestRow[31])

                index = newestIndex - difference

                tday = datetime.datetime.today()
                todaysDate = str(tday.date())
                now = tday.time()
                endTime = now.replace(hour=14, minute=59, second=59)
                startTime = now.replace(hour=8, minute=29, second=0)


                while(index <= newestIndex):

                    newestRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index)
                    previousRow = Database_Management.Get_Data_From_LootLoaderDataBase.Historical.Get_Row_By_Index(symbol, index-1)

                    if(newestRow == "" or previousRow == ""):
                        
                        newest_ema200_value = 0
                        previous_ema200_value = 0

                        newest_hlev2000_UpperArm = 0
                        previous_ehlev2000_UpperArm = 0
                        
                        closePrice = 0
                        previousClosePrice = 0

                        buyPrice = 0

                        dBdate = todaysDate
                        time = now

                        ema1 = 0
                        ema2 = 0
                        ema3 = 0

                    else:

                        newest_ema200_value = float(newestRow[7])
                        previous_ema200_value = float(previousRow[7])
                        
                        newest_hlev2000_UpperArm = float(newestRow[33])
                        previous_hlev2000_UpperArm = float(previousRow[33])
                        
                        hlevOrigin = float(newestRow[30])
                        previous_hlevOrigin = float(previousRow[30])
                        
                        closePrice = float(newestRow[3])
                        previousClosePrice = float(previousRow[3])

                        buyPrice = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_buyPrice_from_DB(symbol)

                        previousEMA1 = Calculation_Management.Calculate_Upper_EMAs(closePrice, previous_ema200_value)[0]
                        previousEMA2 = Calculation_Management.Calculate_Upper_EMAs(closePrice, previous_ema200_value)[1]
                        #previousEMA3 = Calculation_Management.Calculate_Upper_EMAs(closePrice, previous_ema200_value)[2]

                        ema1 = Calculation_Management.Calculate_Upper_EMAs(closePrice, newest_ema200_value)[0]
                        ema2 = Calculation_Management.Calculate_Upper_EMAs(closePrice, newest_ema200_value)[1]
                        #ema3 = Calculation_Management.Calculate_Upper_EMAs(closePrice, newest_ema200_value)[2]

                        dBdate = str(newestRow[1]).replace(" '","")
                        dBdate = dBdate.replace("'","")
                        dBdate = dBdate.replace(" ","")

                        time = str(newestRow[2]).replace(" '","")
                        time = time.replace("'","")
                        time = time.replace(" ","")

                        time = datetime.datetime.strptime(time, '%H:%M:%S')


                    index = index + 1


                    

                    if(closePrice > buyPrice and (startTime < now and now < endTime)):

                        # if the ema200 is between the hlevOrigin and newest_hlev2000_UpperArm
                        # and the closePrice crosses below newest_hlev2000_UpperArm

                        if(newest_ema200_value < hlevOrigin and newest_ema200_value > newest_hlev2000_UpperArm):

                            if((previousClosePrice > newest_hlev2000_UpperArm and closePrice < newest_hlev2000_UpperArm)):

                                accountBalance = Account_Management.Get_Account_Balance_Available_For_Trading()
                                numberOfShares = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_numberOfShares_from_DB(symbol)
                                #put accountBalance into the LLDB
                                

                                #send sell signal to charles schwab
                                #------------------------------------------------------------------------------------------------            
                                accountHash = Account_Management.Get_Account_Hash()
                                auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
                                order = schwab.orders.equities.equity_sell_market(symbol, numberOfShares)
                                r = auth.place_order(accountHash, order)
                                tradeID = schwab.utils.Utils(auth, accountHash).extract_order_id(r)
                                #------------------------------------------------------------------------------------------------                       


                                #put the trades data into the trades database
                                #------------------------------------------------------------------------------------------------                        
                                localDate = dBdate
                                time = str(time).replace("1900-01-01 ","")
                                localTime = time                        
                                lookingForBuy = 0
                                alreadyBought = 0
                                lookingForSell = 0
                                alreadySold = 0
                                orderID = tradeID#Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_orderID_from_DB(symbol)
                                buyPrice = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_buyPrice_from_DB(symbol)
                                orderType = "SELL" 
                                try:
                                    sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID) 
                                except:
                                    tme.sleep(2)
                                    sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID)

                                Profit_or_Loss = (sellPrice - buyPrice)*numberOfShares
                                roi = (sellPrice - buyPrice)/buyPrice
                                if(sellPrice >= buyPrice):
                                    Good_Bad_Trade = "GOOD"
                                if(sellPrice < buyPrice):
                                    Good_Bad_Trade = "BAD"
                                riskAmount = buyPrice * numberOfShares
                                lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                                if(str(lastTradingRow) == "None"):
                                    rowID = 1
                                else:
                                    rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                                incomeTaxesDue = 0.23*Profit_or_Loss
                                
                                Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                        lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                        alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                        riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                        orderID, rowID)
                                #-------------
                                time = str(time).replace("1900-01-01 ","")
                                message = symbol, True, dBdate, time
                                print(message)
                                Logging.info(str(message)+"\r***************************************Symbol {} sold {} shares at {}.**************************************".format(symbol, numberOfShares, sellPrice))
                                
                                # message = "***************************************Symbol {} sold {} shares at {}.**************************************".format(symbol, numberOfShares, sellPrice)
                                # print(message)
                                # Logging.info_Trading(message)
                                return

                        


                        # if the ema200 is between the hlevOrigin and newest_hlev2000_UpperArm
                        # and the closePrice crosses below ema200

                        if(newest_ema200_value < hlevOrigin and newest_ema200_value > newest_hlev2000_UpperArm):

                            if((previousClosePrice > previous_ema200_value and closePrice < newest_ema200_value)):

                                accountBalance = Account_Management.Get_Account_Balance_Available_For_Trading()
                                numberOfShares = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_numberOfShares_from_DB(symbol)
                                #put accountBalance into the LLDB
                                

                                #send buy signal to charles schwab (still need to put STOP value in here)
                                #------------------------------------------------------------------------------------------------            
                                accountHash = Account_Management.Get_Account_Hash()
                                auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
                                order = schwab.orders.equities.equity_sell_market(symbol, numberOfShares)
                                r = auth.place_order(accountHash, order)
                                tradeID = schwab.utils.Utils(auth, accountHash).extract_order_id(r)
                                #------------------------------------------------------------------------------------------------                       


                                #put the trades data into the trades database
                                #------------------------------------------------------------------------------------------------                        
                                localDate = dBdate
                                time = str(time).replace("1900-01-01 ","")
                                localTime = time                        
                                lookingForBuy = 0
                                alreadyBought = 0
                                lookingForSell = 0
                                alreadySold = 0
                                orderID = tradeID#Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_orderID_from_DB(symbol)
                                buyPrice = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_buyPrice_from_DB(symbol)
                                orderType = "SELL" 
                                
                                try:
                                    sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID) 
                                except:
                                    tme.sleep(2)
                                    sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID)

                                Profit_or_Loss = (sellPrice - buyPrice)*numberOfShares
                                roi = (sellPrice - buyPrice)/buyPrice
                                if(sellPrice >= buyPrice):
                                    Good_Bad_Trade = "GOOD"
                                if(sellPrice < buyPrice):
                                    Good_Bad_Trade = "BAD"
                                riskAmount = buyPrice * numberOfShares
                                lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                                if(str(lastTradingRow) == "None"):
                                    rowID = 1
                                else:
                                    rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                                incomeTaxesDue = 0.23*Profit_or_Loss
                                
                                Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                        lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                        alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                        riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                        orderID, rowID)
                                #-------------
                                time = str(time).replace("1900-01-01 ","")
                                message = symbol, True, dBdate, time
                                print(message)
                                Logging.info(str(message)+"\r***************************************Symbol {} sold {} shares at {}.**************************************".format(symbol, numberOfShares, sellPrice))
                                
                                return
                        
                        
                        # if ema200 is less than the origin and the closePrice crosses the origin
                        if(newest_ema200_value < hlevOrigin):
                            if((previousClosePrice > previous_hlevOrigin and closePrice < hlevOrigin)):


                                accountBalance = Account_Management.Get_Account_Balance_Available_For_Trading()
                                numberOfShares = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_numberOfShares_from_DB(symbol)
                                #put accountBalance into the LLDB
                                

                                #send buy signal to charles schwab (still need to put STOP value in here)
                                #------------------------------------------------------------------------------------------------            
                                accountHash = Account_Management.Get_Account_Hash()
                                auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
                                order = schwab.orders.equities.equity_sell_market(symbol, numberOfShares)
                                r = auth.place_order(accountHash, order)
                                tradeID = schwab.utils.Utils(auth, accountHash).extract_order_id(r)
                                #------------------------------------------------------------------------------------------------                       


                                #put the trades data into the trades database
                                #------------------------------------------------------------------------------------------------                        
                                localDate = dBdate
                                time = str(time).replace("1900-01-01 ","")
                                localTime = time                        
                                lookingForBuy = 0
                                alreadyBought = 0
                                lookingForSell = 0
                                alreadySold = 0
                                orderID = tradeID#Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_orderID_from_DB(symbol)
                                buyPrice = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_buyPrice_from_DB(symbol)
                                orderType = "SELL" 
                                
                                try:
                                    sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID) 
                                except:
                                    tme.sleep(2)
                                    sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID)

                                Profit_or_Loss = (sellPrice - buyPrice)*numberOfShares
                                roi = (sellPrice - buyPrice)/buyPrice
                                if(sellPrice >= buyPrice):
                                    Good_Bad_Trade = "GOOD"
                                if(sellPrice < buyPrice):
                                    Good_Bad_Trade = "BAD"
                                riskAmount = buyPrice * numberOfShares
                                lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                                if(str(lastTradingRow) == "None"):
                                    rowID = 1
                                else:
                                    rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                                incomeTaxesDue = 0.23*Profit_or_Loss
                                
                                Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                        lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                        alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                        riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                        orderID, rowID)
                                #-------------
                                time = str(time).replace("1900-01-01 ","")
                                message = symbol, True, dBdate, time
                                print(message)
                                Logging.info(str(message)+"\r***************************************Symbol {} sold {} shares at {}.**************************************".format(symbol, numberOfShares, sellPrice))
                                
                                return
                        
                        
                        #both the closePrice and ema200 are above the hlevOrigin and closePrice crosses below the ema200
                        if(newest_ema200_value > hlevOrigin and closePrice > hlevOrigin):
                           if((previousClosePrice > previous_ema200_value) and (closePrice < newest_ema200_value)):


                            accountBalance = Account_Management.Get_Account_Balance_Available_For_Trading()
                            numberOfShares = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_numberOfShares_from_DB(symbol)
                            #put accountBalance into the LLDB
                            

                            #send buy signal to charles schwab (still need to put STOP value in here)
                            #------------------------------------------------------------------------------------------------            
                            accountHash = Account_Management.Get_Account_Hash()
                            auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
                            order = schwab.orders.equities.equity_sell_market(symbol, numberOfShares)
                            r = auth.place_order(accountHash, order)
                            tradeID = schwab.utils.Utils(auth, accountHash).extract_order_id(r)
                            #------------------------------------------------------------------------------------------------                       


                            #put the trades data into the trades database
                            #------------------------------------------------------------------------------------------------                        
                            localDate = dBdate
                            time = str(time).replace("1900-01-01 ","")
                            localTime = time                        
                            lookingForBuy = 0
                            alreadyBought = 0
                            lookingForSell = 0
                            alreadySold = 0
                            orderID = tradeID#Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_orderID_from_DB(symbol)
                            buyPrice = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_buyPrice_from_DB(symbol)
                            orderType = "SELL" 
                            
                            try:
                                sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID) 
                            except:
                                tme.sleep(2)
                                sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID)

                            Profit_or_Loss = (sellPrice - buyPrice)*numberOfShares
                            roi = (sellPrice - buyPrice)/buyPrice
                            if(sellPrice >= buyPrice):
                                Good_Bad_Trade = "GOOD"
                            if(sellPrice < buyPrice):
                                Good_Bad_Trade = "BAD"
                            riskAmount = buyPrice * numberOfShares
                            lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                            if(str(lastTradingRow) == "None"):
                                rowID = 1
                            else:
                                rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                            incomeTaxesDue = 0.23*Profit_or_Loss
                            
                            Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                    lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                    alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                    riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                    orderID, rowID)
                            #-------------
                            time = str(time).replace("1900-01-01 ","")
                            message = symbol, True, dBdate, time
                            print(message)
                            Logging.info(str(message)+"\r***************************************Symbol {} sold {} shares at {}.**************************************".format(symbol, numberOfShares, sellPrice))
                            
                            return
                        
                        

                        #or closePrice crosses below hlev2000_UpperArm
                        if((previousClosePrice > previous_hlev2000_UpperArm and closePrice < newest_hlev2000_UpperArm)):


                            accountBalance = Account_Management.Get_Account_Balance_Available_For_Trading()
                            numberOfShares = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_numberOfShares_from_DB(symbol)
                            #put accountBalance into the LLDB
                            

                            #send buy signal to charles schwab (still need to put STOP value in here)
                            #------------------------------------------------------------------------------------------------            
                            accountHash = Account_Management.Get_Account_Hash()
                            auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
                            order = schwab.orders.equities.equity_sell_market(symbol, numberOfShares)
                            r = auth.place_order(accountHash, order)
                            tradeID = schwab.utils.Utils(auth, accountHash).extract_order_id(r)
                            #------------------------------------------------------------------------------------------------                       


                            #put the trades data into the trades database
                            #------------------------------------------------------------------------------------------------                        
                            localDate = dBdate
                            time = str(time).replace("1900-01-01 ","")
                            localTime = time                        
                            lookingForBuy = 0
                            alreadyBought = 0
                            lookingForSell = 0
                            alreadySold = 0
                            orderID = tradeID#Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_orderID_from_DB(symbol)
                            buyPrice = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_buyPrice_from_DB(symbol)
                            orderType = "SELL" 
                            
                            try:
                                    sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID) 
                            except:
                                tme.sleep(2)
                                sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID)

                            Profit_or_Loss = (sellPrice - buyPrice)*numberOfShares
                            roi = (sellPrice - buyPrice)/buyPrice
                            if(sellPrice >= buyPrice):
                                Good_Bad_Trade = "GOOD"
                            if(sellPrice < buyPrice):
                                Good_Bad_Trade = "BAD"
                            riskAmount = buyPrice * numberOfShares
                            lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                            if(str(lastTradingRow) == "None"):
                                rowID = 1
                            else:
                                rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                            incomeTaxesDue = 0.23*Profit_or_Loss
                            
                            Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                    lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                    alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                    riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                    orderID, rowID)
                            #-------------
                            time = str(time).replace("1900-01-01 ","")
                            message = symbol, True, dBdate, time
                            print(message)
                            Logging.info(str(message)+"\r***************************************Symbol {} sold {} shares at {}.**************************************".format(symbol, numberOfShares, sellPrice))
                            
                            return

                            #or ema200 crosses below hlev2000_UpperArm
                        
                        #or ema200 > hlev2000_UpperArm
                        if(newest_ema200_value > hlevOrigin):

                            # and closePrice crosses below ema200
                            if(closePrice < ema1 or closePrice < ema2 or closePrice < ema3):


                                accountBalance = Account_Management.Get_Account_Balance_Available_For_Trading()
                                numberOfShares = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_numberOfShares_from_DB(symbol)
                                #put accountBalance into the LLDB
                                

                                #send buy signal to charles schwab (still need to put STOP value in here)
                                #------------------------------------------------------------------------------------------------            
                                accountHash = Account_Management.Get_Account_Hash()
                                auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
                                order = schwab.orders.equities.equity_sell_market(symbol, numberOfShares)
                                r = auth.place_order(accountHash, order)
                                tradeID = schwab.utils.Utils(auth, accountHash).extract_order_id(r)
                                #------------------------------------------------------------------------------------------------                       


                                #put the trades data into the trades database
                                #------------------------------------------------------------------------------------------------                        
                                localDate = dBdate
                                time = str(time).replace("1900-01-01 ","")
                                localTime = time                        
                                lookingForBuy = 0
                                alreadyBought = 0
                                lookingForSell = 0
                                alreadySold = 0
                                orderID = tradeID#Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_orderID_from_DB(symbol)
                                buyPrice = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_buyPrice_from_DB(symbol)
                                orderType = "SELL" 
                                
                                try:
                                    sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID) 
                                except:
                                    tme.sleep(2)
                                    sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID)

                                Profit_or_Loss = (sellPrice - buyPrice)*numberOfShares
                                roi = (sellPrice - buyPrice)/buyPrice
                                if(sellPrice >= buyPrice):
                                    Good_Bad_Trade = "GOOD"
                                if(sellPrice < buyPrice):
                                    Good_Bad_Trade = "BAD"
                                riskAmount = buyPrice * numberOfShares
                                lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                                if(str(lastTradingRow) == "None"):
                                    rowID = 1
                                else:
                                    rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                                incomeTaxesDue = 0.23*Profit_or_Loss
                                
                                Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                        lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                        alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                        riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                        orderID, rowID)
                                #-------------
                                time = str(time).replace("1900-01-01 ","")
                                message = symbol, True, dBdate, time
                                print(message)
                                Logging.info(str(message)+"\r***************************************Symbol {} sold {} shares at {}.**************************************".format(symbol, numberOfShares, sellPrice))
                                
                                return

                    #failed trade: closePrice <= buyPrice - buyPrice * 0.2
                    if((closePrice < (buyPrice - (buyPrice * 0.75))) and (startTime < now and now < endTime)): 

                        #catch the knife - start over and reset the status to lookingForBuy
                        localDate = dBdate
                        time = str(time).replace("1900-01-01 ","")
                        localTime = time                        
                        lookingForBuy = 0
                        alreadyBought = 0
                        lookingForSell = 0
                        alreadySold = 0

                        #sell if fails
                        #------------------------------------------------------------------------------------------------
                        accountBalance = Account_Management.Get_Account_Balance_Available_For_Trading()
                        numberOfShares = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_numberOfShares_from_DB(symbol)
                        #put accountBalance into the LLDB
                        

                        #send buy signal to charles schwab (still need to put STOP value in here)
                        #------------------------------------------------------------------------------------------------            
                        accountHash = Account_Management.Get_Account_Hash()
                        auth = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
                        order = schwab.orders.equities.equity_sell_market(symbol, numberOfShares)
                        r = auth.place_order(accountHash, order)
                        tradeID = schwab.utils.Utils(auth, accountHash).extract_order_id(r)
                        #------------------------------------------------------------------------------------------------                       


                        #put the trades data into the trades database
                        #------------------------------------------------------------------------------------------------                        
                        localDate = dBdate
                        time = str(time).replace("1900-01-01 ","")
                        localTime = time                        
                        lookingForBuy = 0
                        alreadyBought = 0
                        lookingForSell = 0
                        alreadySold = 0
                        orderID = tradeID#Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_orderID_from_DB(symbol)
                        buyPrice = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_buyPrice_from_DB(symbol)
                        orderType = "SELL" 
                        
                        try:
                            sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID) 
                        except:
                            tme.sleep(2)
                            sellPrice = Pull_From_Schwab_Management.Get_BuyPrice_from_orderID(orderID)                                                           
                        Profit_or_Loss = (sellPrice - buyPrice)*numberOfShares
                        roi = (sellPrice - buyPrice)/buyPrice
                        if(sellPrice >= buyPrice):
                            Good_Bad_Trade = "GOOD"
                        if(sellPrice < buyPrice):
                            Good_Bad_Trade = "BAD"
                        riskAmount = buyPrice * numberOfShares
                        lastTradingRow = Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)
                        if(str(lastTradingRow) == "None"):
                            rowID = 1
                        else:
                            rowID = int(Database_Management.Get_Data_From_LootLoaderDataBase.Trading.Get_Last_Row_of_Trading_Tables(symbol)[19])+1

                        incomeTaxesDue = 0.23*Profit_or_Loss
                        
                        Database_Management.Give_Data_To_LootLoaderDataBase.Give_TradeData_To_DB(symbol, localDate, localTime, accountBalance, 
                                                                                                lookingForBuy, alreadyBought, lookingForSell, 
                                                                                                alreadySold, orderType, closePrice, numberOfShares, 
                                                                                                riskAmount, Profit_or_Loss, roi, incomeTaxesDue, buyPrice, sellPrice,Good_Bad_Trade, 
                                                                                                orderID, rowID)
                        #-------------
                        time = str(time).replace("1900-01-01 ","")
                        message = symbol, True, dBdate, time
                        print(message)
                        Logging.info(str(message)+"\r***************************************Symbol {} sold {} shares at {}.**************************************".format(symbol, numberOfShares, sellPrice))
                        
                        return                  
                else:
                    time = str(time).replace("1900-01-01 ","") 
                    message = symbol+" False: {} {}".format(dBdate, time)
                    print(message)
                    Logging.info(message)
                    array.append(False)              
                    array.append(dBdate)              
                    array.append(time) 
                    return array




        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)


    def Close_All_Trades_Now():
        pass


class Pull_From_Schwab_Management:

    def Get_Historical_Minute_Data_from_Schwab(symbol, startDateTime, endDateTime):

        try:

            authorization = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
            priceHistory = authorization.get_price_history_every_minute(symbol, start_datetime=startDateTime, end_datetime=endDateTime, need_extended_hours_data="yes")

            # fileLable = "PriceHistory_RawData_JSON_Error_Investigation"
            # Logging.test(fileLable, str(priceHistory))
            return json.dumps(priceHistory.json(), indent=4)

        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)
            if("Expecting value:" in traceback.format_exc()):
                
                Logging.error(error)
                return "Null"
  
    
    def Get_Historical_Day_Data_from_Schwab(symbol, startDateTime, endDateTime):

        try:

            authorization = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
            priceHistory = authorization.get_price_history_every_day(symbol, start_datetime=startDateTime, end_datetime=endDateTime, need_extended_hours_data="yes")

            #print(priceHistory)

            return json.dumps(priceHistory.json(), indent=4)

        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)


    def Get_the_Highest_MonthlyHigh_From_SchwabData(symbol, nDate):

        try:

            dateArray = []
            #parse todaysDate
            dateArray = str(nDate).split("-")

            nDate = datetime.date(int(dateArray[0]), int(dateArray[1]), int(dateArray[2]))

            previousDate = nDate - datetime.timedelta(days=30)
            nTime = "14:59"

            connection = Database_Management.Create_My_Connection()
            mycursor = connection.cursor()

            mycursor.execute("SELECT MAX(ClosePrice) FROM lootloaderdatabase.{}_historical WHERE(nDate BETWEEN '{}' AND '{}') AND nTime='{}' LIMIT 1".format(symbol,previousDate,nDate, nTime))

            HighPrice = str(mycursor.fetchall())
            HighPrice = HighPrice.replace("[","")
            HighPrice = HighPrice.replace("]","")
            HighPrice = HighPrice.replace("(","")
            HighPrice = HighPrice.replace(")","")
            HighPrice = HighPrice.replace(",","")
            HighPrice = HighPrice.replace(" ","")
            mycursor.close()


            #print("'"+HighPrice+"'")
            #print(type(HighPrice))

            if(HighPrice == "None"):
                return float("0.00")

            if(HighPrice != "None"):
                #print(HighPrice)
                return float(HighPrice)

        
            
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


    def Get_the_Lowest_MonthlyLowh_From_SchwabData(symbol, nDate):
            
        try:
            dateArray = []
            #parse todaysDate
            dateArray = str(nDate).split("-")

            nDate = datetime.date(int(dateArray[0]), int(dateArray[1]), int(dateArray[2]))

            previousDate = nDate - datetime.timedelta(days=30)
            nTime = "14:59"

            connection = Database_Management.Create_My_Connection()
            mycursor = connection.cursor()

            mycursor.execute("SELECT MIN(ClosePrice) FROM lootloaderdatabase.{}_historical WHERE(nDate BETWEEN '{}' AND '{}') AND nTime='{}' LIMIT 1".format(symbol,previousDate,nDate, nTime))

            LowPrice = str(mycursor.fetchall())
            LowPrice = LowPrice.replace("[","")
            LowPrice = LowPrice.replace("]","")
            LowPrice = LowPrice.replace("(","")
            LowPrice = LowPrice.replace(")","")
            LowPrice = LowPrice.replace(",","")
            LowPrice = LowPrice.replace(" ","")
            mycursor.close()


            #print("'"+HighPrice+"'")
            #print(type(HighPrice))

            if(LowPrice == "None"):
                return float("0.00")

            if(LowPrice != "None"):
                #print(LowPrice)
                return float(LowPrice)

        
            
        except:
            error = traceback.format_exc()                                       
            Logging.error(error)


    def Get_Trade_With_Order_ID(orderID):
        try:
            accountHash = Account_Management.Get_Account_Hash()
            authorization = Authentication_Management.Schwab_Wrapper_Authentication.Get_Schwab_Account_Authorization()
            #accountNumber = Account_Management.Get_Account_Number()

            stuff = schwab.client.Client.get_order(authorization, orderID, accountHash)

            return json.loads(json.dumps(stuff.json(), indent=4))
            #return stuff

        except:
            
            error = traceback.format_exc()
            if("An attempt" in error):
                os.system(f"C:\\Users\\"+windowsUsername+f"\\Desktop\\LootLoader\\LootLoader_CharlesSchwab\\Authenticate_Manually.py")
            Logging.error(error)


    def Get_BuyPrice_from_orderID(orderID):

        tradeData = Pull_From_Schwab_Management.Get_Trade_With_Order_ID(orderID)

        for item in tradeData['orderActivityCollection'][0]['executionLegs']:
                               
            #print(item)

            buyPrice = item.get('price')

            #print(buyPrice)

            return float(buyPrice)








