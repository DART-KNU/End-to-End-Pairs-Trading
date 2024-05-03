import json
from pykalman import KalmanFilter
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#******************<한국어 보이게>*********************
# matplotlib library load
import matplotlib.pyplot as plt

# plot 한글 보이게
from matplotlib import rc
rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False
#*****************************************************

# select test cluster
algorithm = "Hierarchical"
test_num = 0

#*******************<데이터 전처리>*********************
train_data = pd.read_csv(r"C:\Users\user\Desktop\논스 페트\Pairs-Trading-from-Beginning-to-End\data\Train_Data.csv", encoding='CP949')
val_data = pd.read_csv(r"C:\Users\user\Desktop\논스 페트\Pairs-Trading-from-Beginning-to-End\data\Validation Data.csv", encoding='CP949')
test_data = pd.read_csv(r"C:\Users\user\Desktop\논스 페트\Pairs-Trading-from-Beginning-to-End\data\Test_Data.csv", encoding='CP949')

train_data.columns = train_data.iloc[0]
train_data = train_data.iloc[1:]
train_data.set_index('Symbol Name', inplace=True)
train_data = train_data.replace(',', '', regex=True).astype(float)

val_data.columns = val_data.iloc[0]
val_data = val_data.iloc[1:]
val_data.set_index('Symbol Name', inplace=True)
val_data = val_data.replace(',', '', regex=True).astype(float)

test_data.columns = test_data.iloc[0]
test_data = test_data.iloc[1:]
test_data.set_index('Symbol Name', inplace=True)
test_data = test_data.replace(',', '', regex=True).astype(float)

total_data = pd.concat([train_data, val_data, test_data])
total_data = total_data.dropna(axis=1)

train_data = total_data.iloc[:2979]
val_data = total_data.iloc[2979:2979+732]
test_data = total_data.iloc[2979+732:]
#*****************************************************

def engle_granger_test(cols):
    y1 = df[cols[0]]
    y2 = df[cols[1]]
    model = OLS(y1,y2).fit()  
    residuals = model.resid  

    adf_test = adfuller(residuals)  
    adf_statistic = adf_test[0]
    critical_values = adf_test[4]
    p_value = adf_test[1]
    result = {
        'columns': cols,
        'adf_statistic': adf_test[0],  # ADF 통계량
        'p_value': adf_test[1],  # p-값
        'critical_values': adf_test[4],  # 임계값
    }
    return result

def get_pair(df):
    import itertools
    all_combinations  = list(itertools.combinations(df.columns, 2))

    results = []  # 결과를 저장할 리스트
    for combination in all_combinations:
        result = engle_granger_test(list(combination))
        results.append(result)

    results_sorted = sorted(results, key=lambda x: x['p_value']) 
    return results_sorted

def asset_selection(sets):    
    first_pair = sets[0]['columns']  # 가장 연관성이 큰 첫 번째 쌍
    asset1 = set()  # 첫 번째 그룹
    asset2 = set()  # 두 번째 그룹

    asset1.add(first_pair[0])  # 첫 번째 요소를 asset1에 추가
    asset2.add(first_pair[1])  # 두 번째 요소를 asset2에 추가

    # 나머지 쌍을 그룹에 배치
    for pair in sets[1:]:
        a, b = pair['columns']

        if a in asset1 and b in asset2:
            continue  # 이미 다른 그룹에 배치된 경우 건너뜀
        elif a in asset2 and b in asset1:
            continue  # 이미 다른 그룹에 배치된 경우 건너뜀
        elif a in asset1 and b not in asset1:
            asset2.add(b)  # a가 asset1에 있으면 b를 asset2에 배치
        elif a in asset2 and b not in asset2:
            asset1.add(b)  # a가 asset2에 있으면 b를 asset1에 배치
        elif b in asset1 and a not in asset1:
            asset2.add(a)  # b가 asset1에 있으면 a를 asset2에 배치
        elif b in asset2 and a not in asset2:
            asset1.add(a)  # b가 asset2에 있으면 a를 asset1에 배치
        elif (a not in asset1 and b not in asset2) or (b not in asset1 and a not in asset2):
            # a, b 모두 기존 그룹에 속하지 않은 경우 서로 다른 그룹에 배치
            asset1.add(a)
            asset2.add(b)
        else:
            sets.pop()
            continue

    # 결과 출력
    print("Asset 1:", list(asset1))  # 첫 번째 그룹의 요소
    print("Asset 2:", list(asset2))  # 두 번째 그룹의 요소
    return asset1, asset2

def select_result(results_sorted):
    selected_results = [
        result for result in results_sorted 
        if result['adf_statistic'] < result['critical_values']['5%']
    ]
    selected_results
    asset1, asset2 = asset_selection(selected_results)
    selected_results
    for result in selected_results:
        asset1_name, asset2_name = result['columns']
        if asset1_name not in asset1 or asset2_name not in asset2:
            # 두 자산의 순서가 리스트 순서와 일치하지 않는 경우, 순서를 바꿉니다.
            result['columns'] = [asset1_name, asset2_name]
    return selected_results, asset1, asset2

def kalman_graph(selected_results, df):
    stat_all = []; spread_all = []
    for result in selected_results:
        tickers = result['columns']
        try:
            obs_mat = sm.add_constant(np.log(df[tickers[0]].replace(0, np.nan).dropna().values), prepend=False)[:, np.newaxis]
            delta = 1e-5
            trans_cov = delta / (1 - delta) * np.eye(2)

            kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                              initial_state_mean=np.zeros(2),
                              initial_state_covariance=np.ones((2, 2)),
                              transition_matrices=np.eye(2),
                              observation_matrices=obs_mat,
                              observation_covariance=1.0,
                              transition_covariance=trans_cov)

            state_means, state_covs = kf.filter(np.log(df[tickers[1]].replace(0, np.nan).dropna().values))
            beta_kf = pd.DataFrame(state_means, columns=['slope', 'intercept'])
            beta_kf.plot(subplots=True)
            plt.show()
            
            colors = np.linspace(0.1, 1, len(df))
            sc = plt.scatter(df[tickers[0]], df[tickers[1]], s=50, c=colors, cmap=plt.get_cmap('jet'), edgecolor='k', alpha=0.7)
            cb = plt.colorbar(sc)
            cb.ax.set_yticks(np.linspace(0, 1, len(df[::len(df)//9])))
            cb.ax.set_yticklabels([p.strftime('%Y-%m-%d') for p in df[::len(df)//9].index])

            xi = np.linspace(df[tickers[0]].min(), df[tickers[0]].max(), 2)
            for i, b in enumerate(state_means[::50]):
                plt.plot(xi, b[0] * xi + b[1], alpha=0.5, lw=2, c=plt.get_cmap('jet')(np.linspace(0.1, 1, len(state_means[::50]))[i]))
            plt.show()

            beta_kf.index = df.index
            spread_kf = df[tickers[1]] - df[tickers[0]] * beta_kf['slope'] - beta_kf['intercept']
            spread_all.append([spread_kf])
            spread_kf.plot(label='kf_spread')

            stat_all.append([tickers, state_means, state_covs])

            plt.ylabel('spread')
            plt.legend()
            plt.show()

        except Exception as e:
            print(f"An error occurred for ticker pair {tickers}: {e}")

    return stat_all, spread_all


def get_spread(selected_results, spread_all):
    total_inverse_p_value = sum(1 / item['p_value'] for item in selected_results)
    for item in selected_results:
        item['ratio'] = (1 / item['p_value']) / total_inverse_p_value
    ratios = np.array([item['ratio'] for item in selected_results])  # 이제 numpy 배열로 변환하여 문제 해결 시도
    spread_series = [pd.Series(spread) for spread in spread_all]
    weighted_spreads = [spread_series[i][0] * ratios[i] for i in range(len(spread_series))]
    weighted_sum_spreads = pd.concat(weighted_spreads, axis=1).sum(axis=1)
    weighted_sum_spreads_series = pd.Series(weighted_sum_spreads)
    weighted_sum_spreads_series.plot()
    plt.title('Sum of Spreads Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weighted Sum of Spreads')
    plt.show()
    return weighted_sum_spreads_series

def adftest(data):
    #adf 검정!
    result = adfuller(data)

    print('정상성이 있는지 판단 : ')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return key

def ARIMA_model(data):
    #델타yt의 경우와 그냥 yt의 경우에 감마의 값에 따라 평균회귀 가능성이 어떻게 달라지는지 조심하세요
    model = ARIMA(data, order=(1, 0, 0))
    model_fit = model.fit()
    print(f'λ 값의 유도(델타yt가 아닌 그냥 yt임 주의!!) : { model_fit.params[1] }')
    half = -np.log(2)/(model_fit.params[1]-1)
    print(f'반감기 : {half}')

    print(model_fit.summary())
    return half

import numpy as np

import numpy as np

def RS_func(data):

    # 데이터를 최소값이 1 이상이 되도록 조정
    offset = 1 - np.min(data[data > 0])  # 0 이상인 값 중 최소값을 1 이상으로 조정
    adjusted_data = data + offset

    # 비율 계산
    ratios = adjusted_data[1:] / adjusted_data[:-1]
    
    # 로그 변환 전 양수 값인지 다시 확인
    positive_ratios = ratios[ratios > 0]
    L = np.log10(positive_ratios)

    if len(L) == 0:
        return 0  # 유효한 로그 값이 없으면 0을 반환

    # 평균 조정된 누적합과 R/S 계산
    Z = np.mean(L)
    C = np.cumsum(L - Z)
    R = max(C) - min(C)
    S = np.std(L)
    if S == 0:
        return 0  # 표준편차가 0이면 나눗셈을 수행하지 않고 0을 반환
    else:
        return R / S



from sklearn.linear_model import LinearRegression

def hurst_func(data):
    if len(data) < 2:
        return 0  # 데이터 포인트가 1개 이하인 경우, 분석할 수 없으므로 0을 반환

    min_window = 10
    max_window = len(data) - 1
    by_factor = np.log10(2.0)
    window_sizes = list(map(lambda x: int(10 ** x), np.arange(np.log10(min_window), np.log10(max_window), by_factor)))
    window_sizes.append(len(data))

    RS = []
    for w in window_sizes:
        rs = []
        for start in range(0, len(data), w):
            if (start + w) > len(data):
                break
            res = RS_func(data[start:start + w].astype(np.float64))
            if res != 0:
                rs.append(res)
        if rs:  # rs가 비어있지 않은 경우에만 평균을 계산
            RS.append(np.mean(rs))

    if not RS or np.any(np.isnan(RS)) or np.any(np.isinf(RS)):
        return 0  # RS가 비어 있거나 계산 불가능한 값이 있으면 0을 반환하여 오류를 방지합니다.

    RS = np.array(RS)
    RS = RS[RS > 0]  # 0 이상인 값만 사용
    if len(RS) == 0:
        return 0

    log_window_sizes = np.log10(window_sizes[:len(RS)])
    log_RS = np.log10(RS)

    lm1 = LinearRegression()
    lm1.fit(log_window_sizes.reshape(-1, 1), log_RS.reshape(-1, 1))
    hurst_exp = lm1.coef_[0][0]
    print(f'H = {hurst_exp}')
    return hurst_exp


# 상태 플래그
def get_signal(weighted_sum_spreads_series,mean_spread, close_range, upper_stop_bound, lowwer_stop_bound,half):
    above = False
    below = False
    keep = False
    signals = []
    startday=0

    for date, value in weighted_sum_spreads_series.items():
        day = weighted_sum_spreads_series.index.get_loc(date)
        if value > upper_bound:
            above = True
        elif value < lower_bound:
            below = True
        elif keep and ((mean_spread - close_range <= value <= mean_spread + close_range) or value > upper_stop_bound or value < lowwer_stop_bound or day - startday > half):
            for index, result in enumerate(selected_results):
                asset1_name = result['columns'][0]
                asset2_name = result['columns'][1]
                if asset1_name in asset1 and asset2_name in asset2:
                    signals.append({
                        'Date': date,
                        'asset1': asset1_name,
                        'asset2': asset2_name,
                        'Investment Amount': result['ratio'],
                        'Investment Type': 'Close'
                    })
                else:
                    signals.append({
                        'Date': date,
                        'asset1': asset2_name,
                        'asset2': asset1_name,
                        'Investment Amount': result['ratio'],
                        'Investment Type': 'Close'
                    })
            keep = False
            below = False
            above = False
        elif (keep == 0) and above and value <= upper_bound:
            for index, result in enumerate(selected_results):
                asset1_name = result['columns'][0]
                asset2_name = result['columns'][1]
                if asset1_name in asset1 and asset2_name in asset2:
                    signals.append({
                        'Date': date,
                        'asset1': asset1_name,
                        'asset2': asset2_name,
                        'Investment Amount': result['ratio'],
                        'Investment Type': 'Long'
                    })
                else:
                    signals.append({
                        'Date': date,
                        'asset1': asset2_name,
                        'asset2': asset1_name,
                        'Investment Amount': result['ratio'],
                        'Investment Type': 'Short'
                    })
            above = False
            keep = True
            startday = weighted_sum_spreads_series.index.get_loc(date)
        elif (keep == 0) and below and value >= lower_bound:
            for index, result in enumerate(selected_results):
                asset1_name = result['columns'][0]
                asset2_name = result['columns'][1]
                if asset1_name in asset1 and asset2_name in asset2:
                    signals.append({
                        'Date': date,
                        'asset1': asset1_name,
                        'asset2': asset2_name,
                        'Investment Amount': result['ratio'],
                        'Investment Type': 'Short'
                    })
                else:
                    signals.append({
                        'Date': date,
                        'asset1': asset2_name,
                        'asset2': asset1_name,
                        'Investment Amount': result['ratio'],
                        'Investment Type': 'Long'
                    })
            below = False
            keep = True
            startday = weighted_sum_spreads_series.index.get_loc(date)

    # 데이터프레임으로 변환
    signals_df = pd.DataFrame(signals)
    return signals_df

def spread_graph(weighted_sum_spreads_series, signals_df):
    # 데이터 타입을 datetime으로 변환 (필요한 경우)
    signals_df['Date'] = pd.to_datetime(signals_df['Date'])
    weighted_sum_spreads_series.index = pd.to_datetime(weighted_sum_spreads_series.index)

    # weighted_sum_spreads_series 그래프 그리기
    plt.figure(figsize=(10, 6))
    weighted_sum_spreads_series.plot()

    # signals_df에서 중복 없이 모든 날짜 가져오기
    unique_dates = signals_df['Date'].drop_duplicates()

    # 각 날짜에 대해 수직선 그리기
    for date in unique_dates:
        plt.axvline(x=date, color='r', linestyle='--', linewidth=0.8)

    # 그래프에 타이틀 및 레이블 추가
    plt.title('Weighted Sum Spreads with Investment Signals')
    plt.xlabel('Date')
    plt.ylabel('Weighted Sum Spreads')
    plt.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound (2 Std Dev)')
    plt.axhline(y=upper_stop_bound, color='r', linestyle='--', label='Upper Bound (2 Std Dev)')
    plt.axhline(y=lower_bound, color='g', linestyle='--', label='Lower Bound (2 Std Dev)')
    plt.axhline(y=lowwer_stop_bound, color='g', linestyle='--', label='Lower Bound (2 Std Dev)')

    # 그래프 보여주기
    plt.show()

with open(r'C:\Users\user\Desktop\논스 페트\Pairs-Trading-from-Beginning-to-End\BTE pairs trading\cluster_data.json', 'r', encoding='utf-8') as f:
    groups = json.load(f)

df = train_data[groups[algorithm][test_num]]
results_sorted = get_pair(df)
selected_results, asset1, asset2 = select_result(results_sorted)
print(selected_results)
stat_all, spread_all = kalman_graph(selected_results, df)
weighted_sum_spreads_series = get_spread(selected_results, spread_all)
key = adftest(weighted_sum_spreads_series)
half=ARIMA_model(weighted_sum_spreads_series)
H = hurst_func(weighted_sum_spreads_series)

#**********************임계점 설정*******************************
mean_spread = weighted_sum_spreads_series.mean()
std_spread = weighted_sum_spreads_series.std()
upper_bound = mean_spread + 2 * std_spread
upper_stop_bound = mean_spread + 3 * std_spread
lower_bound = mean_spread - 2 * std_spread
lowwer_stop_bound = mean_spread - 3 * std_spread
close_range = std_spread * 0.1
print(upper_bound,upper_stop_bound,lower_bound,lowwer_stop_bound)
#***************************************************************

test_df = test_data[groups[algorithm][test_num]]
stat_all, spread_all = kalman_graph(selected_results, test_df)
weighted_sum_spreads_series = get_spread(selected_results, spread_all)
signals_df = get_signal(weighted_sum_spreads_series,mean_spread, close_range, upper_stop_bound, lowwer_stop_bound,half)
print(signals_df)

#********************Backtest*************************************
# 초기 자본 설정
initial_capital = 100000000  # 1억

# 자본 변동을 추적할 DataFrame 준비
capital_df = pd.DataFrame(index=test_df.index)
capital_df['capital'] = initial_capital
signals_df['Date'] = signals_df['Date'].astype(str)
# 포지션을 추적할 딕셔너리
position_all = []

# 각 신호를 처리
for index, signal in signals_df.iterrows():
    date = signal['Date']
    asset1 = signal['asset1']
    asset2 = signal['asset2']
    investment_ratio = signal['Investment Amount']
    trade_type = signal['Investment Type']

    pair_index = [i for i, pairs in enumerate(selected_results) if pairs['columns'] == [asset1, asset2]][0]

    # 자본에서 투자할 금액 계산
    investment_amount = initial_capital * investment_ratio
    hedge_ratios = pd.Series(stat_all[pair_index][1][:, 0], index=pd.Series(spread_all[pair_index][0]).index)
    # 헷지 비율에 따른 투자 금액 계산

    if date in pd.Series(spread_all[pair_index][0]).index:
        hedge_ratio = 1
        amount_asset1 = investment_amount / (1 + hedge_ratio)
        amount_asset2 = amount_asset1 * hedge_ratio
    else:
        continue  # 해당 날짜에 헷지 비율 데이터가 없으면 거래를 진행하지 않음
    
    # 가격 정보
    price_asset1 = test_df.loc[date, asset1]
    price_asset2 = test_df.loc[date, asset2]
    
    # 거래 수량 계산
    quantity_asset1 = amount_asset1 / price_asset1
    quantity_asset2 = amount_asset2 / price_asset2
    positions = {}
    if trade_type == 'Long':
        # Asset1을 매수하고 Asset2를 매도
        positions[date] = {'asset1': (asset1, quantity_asset1, price_asset1,amount_asset1, 'buy'),
                           'asset2': (asset2, quantity_asset2, price_asset2,amount_asset2, 'sell')}
    elif trade_type == 'Short':
        # Asset1을 매도하고 Asset2를 매수
        positions[date] = {'asset1': (asset1, quantity_asset1, price_asset1, amount_asset1,'sell'),
                           'asset2': (asset2, quantity_asset2, price_asset2, amount_asset2,'buy')}
    elif trade_type == 'Close':
        positions[date] = {'asset1': (asset1, quantity_asset1, price_asset1,amount_asset1, 'close'),
                    'asset2': (asset2, quantity_asset2, price_asset2, amount_asset2,'close')}
    position_all.append(positions)

print("\n\n\n\n\n**************BACKTEST*******************\n\n")
print("***************포지션***************8")
print(position_all)

# 데이터프레임으로 변환하기 위한 빈 리스트 생성
data = []

# 각 항목을 반복하면서 데이터 추출 및 정리
for entry in position_all:
    for date, assets in entry.items():
        asset1_data = {
            'date': date,
            'asset_name': assets['asset1'][0],
            'quantity': assets['asset1'][1],
            'price': assets['asset1'][2],
            'amount': assets['asset1'][3],
            'action': assets['asset1'][4],
            'asset_type': 'asset1'
        }
        asset2_data = {
            'date': date,
            'asset_name': assets['asset2'][0],
            'quantity': assets['asset2'][1],
            'price': assets['asset2'][2],
            'amount': assets['asset2'][3],
            'action': assets['asset2'][4],
            'asset_type': 'asset2'
        }
        data.append(asset1_data)
        data.append(asset2_data)

# 데이터 리스트를 데이터프레임으로 변환
position_df = pd.DataFrame(data)

# 데이터 필터링
buy_short_df = position_df[(position_df['action'] == 'buy') | (position_df['action'] == 'sell')]
close_df = position_df[position_df['action'] == 'close']

# 'buy' 및 'short' 데이터에 대한 집계
buy_short_agg = buy_short_df.groupby(['date', 'asset_name', 'action']).agg({
    'quantity': 'sum',  # 총 수량 합계
    'price': 'mean'     # 평균 가격
}).reset_index()

close_agg = close_df.groupby(['date', 'asset_name']).agg({
    'quantity': 'sum',  # 총 수량 합계
    'price': 'mean'     # 평균 가격
}).reset_index()

# 날짜 데이터 타입을 보장하기 위해 변환
buy_short_agg['date'] = pd.to_datetime(buy_short_agg['date'])
close_agg['date'] = pd.to_datetime(close_agg['date'])

# 날짜에 따라 정렬 (merge_asof의 요구사항)
buy_short_agg.sort_values('date', inplace=True)
close_agg.sort_values('date', inplace=True)

# buy_short_agg의 각 행에 대해 date_close 컬럼이 date_trade 이후 가장 가까운 날짜를 갖는 행을 찾아 매칭
matched_df = pd.merge_asof(buy_short_agg, close_agg, on='date', by='asset_name', suffixes=('_trade', '_close'), direction='forward')

# 수익 계산
matched_df['profit'] = matched_df.apply(
    lambda row: row['quantity_trade'] * (row['price_trade'] - row['price_close']) if row['action'] == 'buy'
    else row['quantity_trade'] * (row['price_close'] - row['price_trade']),
    axis=1
)

# 수익 그래프 그리기
plt.figure(figsize=(10, 5))
plt.bar(matched_df['date'].astype(str) + ' ' + matched_df['asset_name'], matched_df['profit'], color='green')
plt.xticks(rotation=45)
plt.title('Profit from Transactions to Close Actions')
plt.xlabel('Date and Asset Name')
plt.ylabel('Profit')
plt.grid(True)
plt.show()

# 날짜별로 수익 합산
daily_profit = matched_df.groupby('date')['profit'].sum().reset_index()

# 시작과 끝 날짜 추가
start_date = pd.DataFrame({'date': [pd.Timestamp('2019-04-01')], 'profit': [0]})
end_date = pd.DataFrame({'date': [pd.Timestamp('2024-03-29')], 'profit': [0]})

# 데이터프레임 병합
daily_profit = pd.concat([start_date, daily_profit, end_date]).reset_index(drop=True)

# 날짜별 누적 수익 계산
daily_profit['cumulative_profit'] = daily_profit['profit'].cumsum()

# 초기 자본금 설정
initial_capital = 100000000  # 1억 원

# 기초자산에 누적 수익 더하기
daily_profit['total_assets'] = initial_capital + daily_profit['cumulative_profit']

# 수익 그래프 그리기
plt.figure(figsize=(12, 6))
plt.step(daily_profit['date'], daily_profit['total_assets'], where='post', label='Total Assets', color='blue')
plt.title('Total Assets Over Time with Step Changes')
plt.xlabel('Date')
plt.ylabel('Total Assets (in 100 millions)')
plt.grid(True)
plt.legend()
plt.show()