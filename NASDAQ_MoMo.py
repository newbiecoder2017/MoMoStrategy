from __future__ import print_function
import pandas as pd
# import pandas_datareader as web
import numpy as np
# import datetime
# from datetime import date
# import Quandl
import matplotlib.pyplot as plt


def read_clean_data():
    """Read the raw price file and create a sample data"""
    try:
        frame = pd.read_hdf("C:/Python27/Examples/wiki_prices.h5", 'table')
        ticker = frame.columns
        test_data = frame[ticker[:2000]]
        test_data = test_data.sort_index(ascending=True)
        test_data.to_hdf("C:/Users/yprasad/Dropbox/NASDAQ/Test_Data.h5", 'table')

    except Exception as e:
        print ("Error Occured in read_clean_data method", e)

def calculate_returns():
    """Read the price file and calculate the returns file"""
    try:
        # price_frame = pd.read_hdf("C:/Users/yprasad/Dropbox/NASDAQ/Test_Data.h5", 'table')

        # price_frame.index = price_frame.index.to_datetime()

        daily_return = pd.read_csv("C:/Python27/Examples/MoMo/DailyReturns.csv", index_col=['Date'], parse_dates=True)

        #         resample_monthly = price_frame.resample('BM').last()

        monthly_return = pd.read_csv("C:/Python27/Examples/MoMo/MonthlyRet.csv", index_col=['Date'], parse_dates=True)

        return daily_return, monthly_return

    except Exception as e:

        print ("Error Occured in calculate_returns method", e)


def generic_momentum(per, samp_per):

    try:

        #         daily_price_data = pd.read_hdf("C:/Users/yprasad/Dropbox/NASDAQ/Test_Data.h5", 'table')

        #         daily_price_data.dropna(how='all', inplace=True)

        #         monthly_price_data = daily_price_data.resample(samp_per).last()  # convert daily to business month end

        monthly_return = pd.read_csv("C:/Python27/Examples/MoMo/MonthlyRet.csv", index_col=['Date'], parse_dates=True)

        generic_gross_returns = monthly_return + 1.0  # calculate the gross monthly return

        generic_rolling_cagr = generic_gross_returns.rolling(per).apply(lambda x: x[:-1].prod() - 1.)

        # generic_rolling_cagr = generic_gross_returns.rolling(per).apply(lambda x: x.prod() - 1.)  # calculate the 12 month CAGR based on the CAGR

        generic_rolling_cagr.to_csv("C:/Python27/Examples/MoMo/Roll_MonthlyRet.csv")  # save the CAGR returns

        return generic_rolling_cagr

    except Exception as e:

        print("Error Occured in generic_momentum method", e)



def calculate_daily_fip(generic_roll_ret, fip_window, samp_per):
    try:

        #         price_data_daily = pd.read_hdf("C:/Users/yprasad/Dropbox/NASDAQ/Test_Data.h5", 'table')  # read the price file

        #         price_data_daily.dropna(how='all', inplace=True)

        # daily_return = price_data_daily.pct_change()  # calculate the daily price returns

        daily_return = pd.read_csv("C:/Python27/Examples/MoMo/DailyReturns.csv", index_col=['Date'], parse_dates=True)

        s1 = daily_return[daily_return >= 0.0].rolling(window=fip_window).count()
        s2 = daily_return[daily_return < 0.0].rolling(window=fip_window).count()

        t1 = s1/fip_window
        t2 = s2/fip_window

        temp = t2.subtract(t1)

        generic_roll_ret[generic_roll_ret >= 0] = 1  # calculate the +ve or -negative sign from the 12month CAGR

        generic_roll_ret[generic_roll_ret < 0] = -1

        #         generic_roll_ret.csv("C:/Python27/Examples/MoMo/DailyReturns.csv")

        daily_fip_to_monthly = temp.resample(samp_per).last()  # convert daily fip to BM end fip


        signed_monthly_fip = daily_fip_to_monthly.multiply(generic_roll_ret)  # multiply monthly FIP to Sign of monthly CAGR return

        signed_monthly_fip.to_csv("C:/Python27/Examples/MoMo/Signed_Monthly_FIP.csv")

        return signed_monthly_fip

    except Exception as e:
        "Error Occured in calculate_daily_fip method"


def cagr_fip_filter_holdings(t1=0.8, t2=0.9, bucket='m'):
    try:
        if bucket == 'm':
            monthly_return = pd.read_csv("C:/Python27/Examples/MoMo/MonthlyRet.csv", index_col=['Date'], parse_dates=True)

            monthly_roll_return = pd.read_csv("C:/Python27/Examples/MoMo/Roll_MonthlyRet.csv", index_col=['Date'], parse_dates=True)

            sign_monthly_fip = pd.read_csv("C:/Python27/Examples/MoMo/Signed_Monthly_FIP.csv", index_col=['Date'], parse_dates=True)

            # sign_monthly_fip = sign_monthly_fip[:-1]

            monthly_roll_return['Lower'] = monthly_roll_return.quantile(t1, axis=1)

            monthly_roll_return['Upper'] = monthly_roll_return.quantile(t2, axis=1)

            signed_cagr_filter = pd.DataFrame({x: np.where(((monthly_roll_return[x] >= monthly_roll_return['Lower']) & (monthly_roll_return[x] <= monthly_roll_return['Upper'])), sign_monthly_fip[x], np.nan)
                                               for x in sign_monthly_fip.columns}, index = monthly_roll_return.index)

            signed_cagr_filter.to_csv("C:/Python27/Examples/MoMo/Monthly/Signed_CAGR_Filter.csv")

            signed_cagr_filter['PerFilter'] = signed_cagr_filter.quantile(0.5, axis = 1)

            cagr_fip_filter = pd.DataFrame({x: np.where(signed_cagr_filter[x] <= signed_cagr_filter['PerFilter'], monthly_return[x], None) for x in monthly_return.columns}, index=monthly_return.index)

            cagr_fip_filter.to_csv("C:/Python27/Examples/MoMo/Monthly/CAGR_FIP_Filter.csv")

            model_trades = pd.DataFrame({i: np.where(cagr_fip_filter[i].notnull(), i, None) for i in cagr_fip_filter.columns}, index=cagr_fip_filter.index)

            model_trades.to_csv("C:/Python27/Examples/MoMo/Monthly/Monthly_Trades.csv")

            returns_frame = pd.DataFrame({x: np.where(model_trades[x].shift(1).notnull(), monthly_return[x], None) for x in model_trades.columns}, index = model_trades.index)

            returns_frame['Port_Ret'] = returns_frame.mean(axis=1, skipna=True) * 100.0

            returns_frame.to_csv("C:/Python27/Examples/MoMo/Monthly/Backtest_Returns.csv")
            # portfolio_return = cagr_fip_filter.mean(axis=1, skipna=True)
            return returns_frame['Port_Ret']

        elif bucket == 'q':

            price_frame = pd.read_csv("C:/Users/yprasad/Dropbox/NASDAQ/NAS_Daily_Price.csv", index_col = ['Date'], parse_dates = True)

            qprice = price_frame.resample('BQ').last()
            qret = qprice.pct_change()

            # qprice = price_frame.resample('BQ').last()
            # qret = qprice.pct_change()
            qret = qret[:-1]

            monthly_return = pd.read_csv("C:/Python27/Examples/MoMo/MonthlyRet.csv", index_col=['Date'], parse_dates=True)

            monthly_roll_return = pd.read_csv("C:/Python27/Examples/MoMo/Roll_MonthlyRet.csv", index_col=['Date'], parse_dates=True)

            sign_monthly_fip = pd.read_csv("C:/Python27/Examples/MoMo/Signed_Monthly_FIP.csv", index_col=['Date'], parse_dates=True)

            # sign_monthly_fip = sign_monthly_fip

            monthly_roll_return['Lower'] = monthly_roll_return.quantile(t1, axis=1)

            monthly_roll_return['Upper'] = monthly_roll_return.quantile(t2, axis=1)

            sign_monthly_fip = sign_monthly_fip.asfreq('BQ', how='last')

            # sign_monthly_fip = sign_monthly_fip[:-1]

            monthly_roll_return = monthly_roll_return.asfreq('BQ', how='last')


            # sign_monthly_fip = sign_monthly_fip.resample('BQ').last()
            #
            # sign_monthly_fip = sign_monthly_fip[:-1]

            # monthly_roll_return = monthly_roll_return.resample('BQ').last()

            signed_cagr_filter = pd.DataFrame({x: np.where(((monthly_roll_return[x] >= monthly_roll_return['Lower']) & (monthly_roll_return[x] <= monthly_roll_return['Upper'])),
                                                           sign_monthly_fip[x], np.nan) for x in sign_monthly_fip.columns}, index = sign_monthly_fip.index)

            signed_cagr_filter.to_csv("C:/Python27/Examples/MoMo/Quaterly/Signed_CAGR_Filter.csv")

            signed_cagr_filter['PerFilter'] = signed_cagr_filter.quantile(0.5, axis=1)

            cagr_fip_filter = pd.DataFrame({x: np.where(signed_cagr_filter[x] <= signed_cagr_filter['PerFilter'], qret[x], None) for x in qret.columns}, index = qret.index)

            cagr_fip_filter.to_csv("C:/Python27/Examples/MoMo/Quaterly/CAGR_FIP_Filter.csv")

            model_trades = pd.DataFrame({i: np.where(cagr_fip_filter[i].notnull(), i, None) for i in cagr_fip_filter.columns}, index=cagr_fip_filter.index)

            model_trades.to_csv("C:/Python27/Examples/MoMo/Quaterly/Monthly_Trades.csv")

            returns_frame = pd.DataFrame({x: np.where(model_trades[x].shift(1).notnull(), qret[x], None) for x in model_trades.columns}, index = model_trades.index)

            returns_frame['Port_Ret'] = returns_frame.mean(axis=1, skipna=True) * 100.0

            returns_frame.to_csv("C:/Python27/Examples/MoMo/Quaterly/Backtest_Returns.csv")

            return returns_frame['Port_Ret']


        elif bucket == 'qo':

            price_frame = pd.read_csv("C:/Users/yprasad/Dropbox/NASDAQ/NAS_Daily_Price.csv", index_col = ['Date'], parse_dates = True)

            qoprice = price_frame.resample('BQ-FEB').last()
            qoret = qoprice.pct_change()

            monthly_return = pd.read_csv("C:/Python27/Examples/MoMo/MonthlyRet.csv", index_col=['Date'], parse_dates=True)

            monthly_roll_return = pd.read_csv("C:/Python27/Examples/MoMo/Roll_MonthlyRet.csv", index_col=['Date'], parse_dates=True)

            sign_monthly_fip = pd.read_csv("C:/Python27/Examples/MoMo/Signed_Monthly_FIP.csv", index_col=['Date'], parse_dates=True)

            monthly_roll_return['Lower'] = monthly_roll_return.quantile(t1, axis=1)

            monthly_roll_return['Upper'] = monthly_roll_return.quantile(t2, axis=1)

            sign_monthly_fip = sign_monthly_fip.asfreq('BQ-FEB', how = 'last')

            monthly_roll_return = monthly_roll_return.asfreq('BQ-FEB', how = 'last')

            signed_cagr_filter = pd.DataFrame({x: np.where(((monthly_roll_return[x] >= monthly_roll_return['Lower']) & (monthly_roll_return[x] <= monthly_roll_return['Upper'])),
                                                           sign_monthly_fip[x], np.nan) for x in sign_monthly_fip.columns}, index = sign_monthly_fip.index)

            signed_cagr_filter.to_csv("C:/Python27/Examples/MoMo/Qtly_Offset/Signed_CAGR_Filter.csv")

            signed_cagr_filter['PerFilter'] = signed_cagr_filter.quantile(0.5, axis=1)

            cagr_fip_filter = pd.DataFrame({x: np.where(signed_cagr_filter[x] <= signed_cagr_filter['PerFilter'], qoret[x], None) for x in qoret.columns}, index=qoret.index)

            cagr_fip_filter.to_csv("C:/Python27/Examples/MoMo/Qtly_Offset/CAGR_FIP_Filter.csv")

            model_trades = pd.DataFrame({i: np.where(cagr_fip_filter[i].notnull(), i, None) for i in cagr_fip_filter.columns}, index=cagr_fip_filter.index)
            '''Save the trades for each decile'''
            model_trades.to_csv("C:/Python27/Examples/MoMo/Qtly_Offset/Monthly_Trades_Quartile_"+str(t1)+".csv")
            '''Print the last rebalance trade recommendation'''
            # print (model_trades[-1:].notnull())

            '''Print the trades for any decile'''
            # if t1 == 0.9:
            #     for i in range(len(model_trades)):
            #         print([col for col in model_trades.columns if model_trades[i:i+1][col].notnull().any()])


            returns_frame = pd.DataFrame({x: np.where(model_trades[x].shift(1).notnull(), qoret[x], None) for x in model_trades.columns}, index = model_trades.index)

            returns_frame['Port_Ret'] = returns_frame.mean(axis=1, skipna=True) * 100.0

            returns_frame.to_csv("C:/Python27/Examples/MoMo/Qtly_Offset/Backtest_Returns.csv")

            return returns_frame['Port_Ret']


    except Exception as e:
        "Error Occured in calculate_daily_fip method", e



# read_clean_data()
# rdaily = calculate_returns()[0]
# rmonthly = calculate_returns()[1]
#
# monthly_rolling_returns = generic_momentum(12, "BM")
#
# calculate_daily_fip(monthly_rolling_returns, 250, 'BM')


mod = input("Enter the rebalance period: ")
mr = pd.read_csv("C:/Python27/Examples/MoMo/MonthlyRet.csv", index_col=['Date'], parse_dates=True)
# mr = mr.resample('BQ-FEB').last()
# idnx = mr.index
# comp_data = pd.DataFrame({x: cagr_fip_filter_holdings(x, x + .1, mod) for x in np.arange(0.0,1.0, 0.1)}, index = idnx)
# cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
# comp_data.columns = cols

if mod =='m':
    mr = mr.resample('BM').last()
    idnx = mr.index
    comp_data = pd.DataFrame({x: cagr_fip_filter_holdings(x, x + .1, mod) for x in np.arange(0.0, 1.0, 0.1)},index=idnx)
    cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
    comp_data.columns = cols
    comp_data.to_csv("C:/Python27/Examples/MoMo/Monthly/Quantile.csv")



elif mod == 'q':
    mr = mr.resample('BQ').last()
    idnx = mr.index
    comp_data = pd.DataFrame({x: cagr_fip_filter_holdings(x, x + .1, mod) for x in np.arange(0.0, 1.0, 0.1)}, index=idnx)
    cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
    comp_data.columns = cols
    comp_data.to_csv("C:/Python27/Examples/MoMo/Quaterly/Quantile.csv")

else:
    mr = mr.resample('BQ-FEB').last()
    idnx = mr.index
    comp_data = pd.DataFrame({x: cagr_fip_filter_holdings(x, x + .1, mod) for x in np.arange(0.0, 1.0, 0.1)},index=idnx)
    cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
    comp_data.columns = cols
    comp_data.to_csv("C:/Python27/Examples/MoMo/Qtly_Offset/Quantile.csv")

comp_data.cumsum().plot()
plt.grid()
plt.show()