from __future__ import print_function
import pandas as pd
import quandl
import numpy as np
quandl.ApiConfig.api_key = 'ziz-Hh4LoVCUV1xT58Se'

import matplotlib.pyplot as plt

def quandl_to_csv():
    '''Identifies the new names in the index, calls for the OHLC prices from quandl and saves it as a csv file'''
    del_sym = []
    add_sym = []

    universe = pd.read_csv("C:/Python27/Git/MoMoStrategy/QQQ_holdings.csv")
    symbols = universe['HoldingsTicker'].tolist()
    new_symbols = sorted([s.strip() for s in symbols])

    uni = pd.read_csv("C:/Python27/Git/MomoStrategy/DailyReturns.csv")
    t = uni.columns.tolist()
    old_symbols = sorted([s.strip() for s in t[1:]])

    for s in old_symbols:
        if s in new_symbols:
            pass
        else:
            del_sym.append(s)

    for s in new_symbols:
        if s in old_symbols:
            pass
        else:
            add_sym.append(s)

    with open("C:/Python27/Git/MoMoStrategy/log.txt",'w') as f:
        f.write("Delete following names: ")
        for d in del_sym:
            f.write(d)
            f.write(',')
        f.write('\n')
        f.write('\n')
        f.write("Add following names: ")

        for a in add_sym:
            f.write(a)
            f.write(',')
    f.close()

    data = quandl.get_table('WIKI/PRICES', ticker = new_symbols,
                            date = { 'gte': '2000-12-31', 'lte': '2018-05-14'},
                            qopts = { 'columns': ['ticker', 'date','adj_open','adj_high','adj_low','close','adj_close']}, paginate=True)

    # create a new dataframe with 'date' column as index
    new = data.set_index('date')

    # use pandas pivot function to sort adj_close by tickers
    clean_data = new.pivot(columns='ticker')

    clean_data.adj_open.to_csv("C:/Python27/Git/MoMoStrategy/a_open.csv")
    clean_data.adj_high.to_csv("C:/Python27/Git/MoMoStrategy/a_high.csv")
    clean_data.adj_low.to_csv("C:/Python27/Git/MoMoStrategy/a_low.csv")
    clean_data.close.to_csv("C:/Python27/Git/MoMoStrategy/close.csv")
    clean_data.adj_close.to_csv("C:/Python27/Git/MoMoStrategy/adj_close.csv")

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

        #this gets the returns from the excel file NASDAQ
        # daily_return = pd.read_csv("C:/Python27/Git/MoMoStrategy/DailyReturns.csv", index_col=['Date'], parse_dates=True)
        daily_prices = pd.read_csv("C:/Python27/Git/MoMoStrategy/adj_close.csv", index_col=['date'], parse_dates=True)
        daily_prices.ffill(inplace = True)
        daily_return= daily_prices.pct_change()
          # this gets the returns from the excel file NASDAQ
        # monthly_return = pd.read_csv("C:/Python27/Git/MoMoStrategy/MonthlyRet.csv", index_col=['Date'], parse_dates=True)
        monthly_prices = daily_prices.resample('BM').last()
        monthly_return = monthly_prices.pct_change()

        return daily_return, monthly_return

    except Exception as e:

        print ("Error Occured in calculate_returns method", e)

def generic_momentum(monthly_return, per, samp_per):

    try:

        generic_gross_returns = monthly_return + 1.0  # calculate the gross monthly return

        generic_rolling_cagr = generic_gross_returns.rolling(per).apply(lambda x: x[:-1].prod() - 1.)

        generic_rolling_cagr.to_csv("C:/Python27/Git/MoMoStrategy/Roll_MonthlyRet.csv")  # save the CAGR returns

        return generic_rolling_cagr

    except Exception as e:

        print("Error Occured in generic_momentum method", e)

def calculate_daily_fip(daily_return, generic_roll_ret, fip_window, samp_per):
    try:

        s1 = daily_return[daily_return >= 0.0].rolling(window=fip_window).count()
        s2 = daily_return[daily_return < 0.0].rolling(window=fip_window).count()

        t1 = s1/fip_window
        t2 = s2/fip_window

        temp = t2.subtract(t1)

        generic_roll_ret[generic_roll_ret >= 0] = 1  # calculate the +ve or -negative sign from the 12month CAGR

        generic_roll_ret[generic_roll_ret < 0] = -1

        daily_fip_to_monthly = temp.resample(samp_per).last()  # convert daily fip to BM end fip

        daily_fip_to_monthly.index = generic_roll_ret.index
        signed_monthly_fip = daily_fip_to_monthly.multiply(generic_roll_ret)  # multiply monthly FIP to Sign of monthly CAGR return

        signed_monthly_fip.to_csv("C:/Python27/Git/MoMoStrategy/Signed_Monthly_FIP.csv")

        return signed_monthly_fip

    except Exception as e:
        "Error Occured in calculate_daily_fip method"

def cagr_fip_filter_holdings(returns_df, t1=0.8, t2=0.9, bucket='m'):
    try:
        if bucket == 'm':

            monthly_return = returns_df

            monthly_roll_return = pd.read_csv("C:/Python27/Git/MoMoStrategy/Roll_MonthlyRet.csv", index_col=['date'], parse_dates=True)

            sign_monthly_fip = pd.read_csv("C:/Python27/Git/MoMoStrategy/Signed_Monthly_FIP.csv", index_col=['date'], parse_dates=True)

            monthly_roll_return['Lower'] = monthly_roll_return.quantile(t1, axis=1)

            monthly_roll_return['Upper'] = monthly_roll_return.quantile(t2, axis=1)

            signed_cagr_filter = pd.DataFrame({x: np.where(((monthly_roll_return[x] >= monthly_roll_return['Lower']) & (monthly_roll_return[x] <= monthly_roll_return['Upper'])), sign_monthly_fip[x], np.nan)
                                               for x in sign_monthly_fip.columns}, index = monthly_roll_return.index)

            signed_cagr_filter.to_csv("C:/Python27/Git/MoMoStrategy/Monthly_Signed_CAGR_Filter.csv")

            signed_cagr_filter['PerFilter'] = signed_cagr_filter.quantile(0.5, axis = 1)

            cagr_fip_filter = pd.DataFrame({x: np.where(signed_cagr_filter[x] <= signed_cagr_filter['PerFilter'], monthly_return[x], None) for x in monthly_return.columns}, index=monthly_return.index)

            cagr_fip_filter.to_csv("C:/Python27/Git/MoMoStrategy/Monthly_CAGR_FIP_Filter.csv")

            model_trades = pd.DataFrame({i: np.where(cagr_fip_filter[i].notnull(), i, None) for i in cagr_fip_filter.columns}, index=cagr_fip_filter.index)

            model_trades.to_csv("C:/Python27/Git/MoMoStrategy/Monthly_Trades.csv")

            returns_frame = pd.DataFrame({x: np.where(model_trades[x].shift(1).notnull(), monthly_return[x], None) for x in model_trades.columns}, index = model_trades.index)

            returns_frame['Port_Ret'] = returns_frame.mean(axis=1, skipna=True) * 100.0

            returns_frame.to_csv("C:/Python27/Git/MoMoStrategy/Monthly_Backtest_Returns.csv")

            return returns_frame['Port_Ret']

        elif bucket == 'q':

            qret = returns_df

            monthly_roll_return = pd.read_csv("C:/Python27/Git/MoMoStrategy/Roll_MonthlyRet.csv", index_col=['date'], parse_dates=True)
            sign_monthly_fip = pd.read_csv("C:/Python27/Git/MoMoStrategy/Signed_Monthly_FIP.csv", index_col=['date'], parse_dates=True)

            monthly_roll_return['Lower'] = monthly_roll_return.quantile(t1, axis=1)

            monthly_roll_return['Upper'] = monthly_roll_return.quantile(t2, axis=1)

            sign_monthly_fip = sign_monthly_fip.asfreq('BQ', how='last')

            monthly_roll_return = monthly_roll_return.asfreq('BQ', how='last')

            signed_cagr_filter = pd.DataFrame({x: np.where(((monthly_roll_return[x] >= monthly_roll_return['Lower']) & (monthly_roll_return[x] <= monthly_roll_return['Upper'])),
                                                           sign_monthly_fip[x], np.nan) for x in sign_monthly_fip.columns}, index = sign_monthly_fip.index)

            signed_cagr_filter.to_csv("C:/Python27/Git/MoMoStrategy/Quaterly_Signed_CAGR_Filter.csv")

            signed_cagr_filter['PerFilter'] = signed_cagr_filter.quantile(0.5, axis=1)

            cagr_fip_filter = pd.DataFrame({x: np.where(signed_cagr_filter[x] <= signed_cagr_filter['PerFilter'], qret[x], None) for x in qret.columns}, index = qret.index)

            cagr_fip_filter.to_csv("C:/Python27/Git/MoMoStrategy/Quaterly_CAGR_FIP_Filter.csv")

            model_trades = pd.DataFrame({i: np.where(cagr_fip_filter[i].notnull(), i, None) for i in cagr_fip_filter.columns}, index=cagr_fip_filter.index)

            model_trades.to_csv("C:/Python27/Git/MoMoStrategy/Quaterly_Monthly_Trades.csv")

            returns_frame = pd.DataFrame({x: np.where(model_trades[x].shift(1).notnull(), qret[x], None) for x in model_trades.columns}, index = model_trades.index)

            returns_frame['Port_Ret'] = returns_frame.mean(axis=1, skipna=True) * 100.0

            returns_frame.to_csv("C:/Python27/Git/MoMoStrategy/Quaterly_Backtest_Returns.csv")

            return returns_frame['Port_Ret']


        elif bucket == 'qo':


            qoret = returns_df

            monthly_roll_return = pd.read_csv("C:/Python27/Git/MoMoStrategy/Roll_MonthlyRet.csv", index_col=['date'], parse_dates=True)
            sign_monthly_fip = pd.read_csv("C:/Python27/Git/MoMoStrategy/Signed_Monthly_FIP.csv", index_col=['date'], parse_dates=True)

            monthly_roll_return['Lower'] = monthly_roll_return.quantile(t1, axis=1)

            monthly_roll_return['Upper'] = monthly_roll_return.quantile(t2, axis=1)


            sign_monthly_fip = sign_monthly_fip.asfreq('BQ-FEB', how = 'last')

            monthly_roll_return = monthly_roll_return.asfreq('BQ-FEB', how = 'last')

            signed_cagr_filter = pd.DataFrame({x: np.where(((monthly_roll_return[x] >= monthly_roll_return['Lower']) & (monthly_roll_return[x] <= monthly_roll_return['Upper'])),
                                                           sign_monthly_fip[x], np.nan) for x in sign_monthly_fip.columns}, index = sign_monthly_fip.index)

            signed_cagr_filter.to_csv("C:/Python27/Git/MoMoStrategy/Qtly_Offset_Signed_CAGR_Filter.csv")

            signed_cagr_filter['PerFilter'] = signed_cagr_filter.quantile(0.5, axis=1)

            qoret = qoret[:-1]

            cagr_fip_filter = pd.DataFrame({x: np.where(signed_cagr_filter[x] <= signed_cagr_filter['PerFilter'], qoret[x], None) for x in qoret.columns}, index=qoret.index)

            cagr_fip_filter.to_csv("C:/Python27/Git/MoMoStrategy/Qtly_Offset_CAGR_FIP_Filter.csv")

            model_trades = pd.DataFrame({i: np.where(cagr_fip_filter[i].notnull(), i, None) for i in cagr_fip_filter.columns}, index=cagr_fip_filter.index)

            '''Save the trades for each decile'''
            model_trades.to_csv("C:/Python27/Git/MoMoStrategy/Qtly_Offset_Monthly_Trades_Quartile_"+str(t1)+".csv")
            '''Print the last rebalance trade recommendation'''
            # print (model_trades[-1:].notnull())

            '''Print the trades for any decile'''
            # if t1 == 0.9:
            #     for i in range(len(model_trades)):
            #         print([col for col in model_trades.columns if model_trades[i:i+1][col].notnull().any()])


            returns_frame = pd.DataFrame({x: np.where(model_trades[x].shift(1).notnull(), qoret[x], None) for x in model_trades.columns}, index = model_trades.index)

            returns_frame['Port_Ret'] = returns_frame.mean(axis=1, skipna=True) * 100.0

            returns_frame.to_csv("C:/Python27/Git/MoMoStrategy/Qtly_Offset_Backtest_Returns.csv")

            return returns_frame['Port_Ret']


    except Exception as e:
        "Error Occured in calculate_daily_fip method", e


if __name__ == "__main__":

    # read_clean_data()
    # quandl_to_csv()

    daily_prices = pd.read_csv("C:/Python27/Git/MoMoStrategy/adj_close.csv", index_col=['date'], parse_dates=True)
    daily_prices.ffill(inplace=True)
    daily_return = daily_prices.pct_change()

    monthly_return = daily_prices.resample('BM').last().ffill()
    monthly_return = monthly_return.pct_change()

    qtr_return = daily_prices.resample('BQ').last().ffill()
    qtr_return = qtr_return.pct_change()

    qtr_return_off = daily_prices.resample('BQ-FEB').last().ffill()
    qtr_return_off = qtr_return_off.pct_change()

    rdaily = calculate_returns()[0]
    rmonthly = calculate_returns()[1]

    monthly_rolling_returns = generic_momentum(monthly_return, 12, "BM")

    signed_monthly_fip = calculate_daily_fip(daily_return, monthly_rolling_returns, 250, 'BM')

    mod = input("Enter the rebalance period: ")

    if mod == 'm':

        idnx = monthly_return.index
        comp_data = pd.DataFrame(
            {x: cagr_fip_filter_holdings(monthly_return, x, x + .1, mod) for x in np.arange(0.0, 1.0, 0.1)},
            index=idnx)
        cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
        comp_data.columns = cols
        print(comp_data)
        # comp_data.to_csv("C:/Python27/Git/MoMoStrategy/Monthly_Quantile.csv")

    elif mod == 'q':

        idnx = qtr_return.index
        comp_data = pd.DataFrame({x: cagr_fip_filter_holdings(qtr_return, x, x + .1, mod) for x in np.arange(0.0, 1.0, 0.1)}, index=idnx)

        cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
        comp_data.columns = cols
        comp_data.to_csv("C:/Python27/Git/MoMoStrategy/Qtr_Quantile.csv")
        print(comp_data)

    elif mod == 'qo':
        idnx = qtr_return_off.index
        comp_data = pd.DataFrame({x: cagr_fip_filter_holdings(qtr_return_off, x, x + .1, mod) for x in np.arange(0.0, 1.0, 0.1)}, index=idnx)
        cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
        comp_data.columns = cols
        comp_data.to_csv("C:/Python27/Git/MoMoStrategy/Qtly_Offset_Quantile.csv")
        print(comp_data)

    ls = comp_data.mean()
    ls_q = [ls['Q10']-x for x in ls]
    plt.plot(ls_q)
    plt.show()


