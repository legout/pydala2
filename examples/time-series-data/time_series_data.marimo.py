import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def __():
    import os
    import tempfile
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.dataset as pds
    from pathlib import Path
    from datetime import datetime, timedelta
    from typing import Dict, Any, List, Optional, Union, Tuple
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from scipy import stats
    import time

    # Import PyDala2 components
    from pydala.dataset import ParquetDataset
    from pydala.table import PydalaTable
    from pydala.catalog import Catalog

    # Set up plotting
    plt.style.use('seaborn-v0_8')
    return Any, ARIMA, Catalog, List, Optional, ParquetDataset, Path, PydalaTable, Tuple, Union, datetime, os, pa, pd, pds, tempdir, timedelta, time, np, plt, sns, seasonal_decompose, stats


@app.cell
def __(np, pd, timedelta, datetime):
    def create_sample_time_series_data():
        """Create comprehensive time series data for analysis."""
        np.random.seed(42)

        # Generate base timestamps
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

        # Create multiple time series with different characteristics
        n_records = len(timestamps)

        # Base trend components
        trend = np.linspace(100, 300, n_records) + np.random.normal(0, 5, n_records)

        # Seasonal components
        daily_seasonality = 10 * np.sin(2 * np.pi * np.arange(n_records) / 24)
        weekly_seasonality = 20 * np.sin(2 * np.pi * np.arange(n_records) / (24 * 7))
        yearly_seasonality = 30 * np.sin(2 * np.pi * np.arange(n_records) / (24 * 365.25))

        # Combine components
        base_value = trend + daily_seasonality + weekly_seasonality + yearly_seasonality

        # Create multiple metrics
        data = {
            'timestamp': timestamps,
            'metric_cpu': np.maximum(0, base_value + np.random.normal(0, 15, n_records)),
            'metric_memory': np.maximum(0, base_value * 0.8 + np.random.normal(0, 12, n_records)),
            'metric_disk_io': np.maximum(0, base_value * 0.3 + np.random.normal(0, 8, n_records)),
            'metric_network': np.maximum(0, base_value * 0.6 + np.random.normal(0, 10, n_records)),
            'metric_errors': np.random.poisson(2, n_records),
            'metric_requests': np.maximum(0, base_value * 1.2 + np.random.normal(0, 20, n_records)),
            'service_name': np.random.choice(['web-server', 'database', 'cache', 'api-gateway'], n_records),
            'environment': np.random.choice(['production', 'staging', 'development'], n_records, p=[0.7, 0.2, 0.1]),
            'region': np.random.choice(['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'], n_records),
            'version': np.random.choice(['v1.0', 'v1.1', 'v1.2', 'v2.0'], n_records, p=[0.4, 0.3, 0.2, 0.1])
        }

        df = pd.DataFrame(data)

        # Add derived time-based columns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_business_hours'] = df['hour'].between(9, 17)

        # Add some anomaly patterns
        anomaly_indices = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
        for idx in anomaly_indices:
            df.loc[idx, 'metric_cpu'] *= np.random.uniform(1.5, 3.0)
            df.loc[idx, 'metric_memory'] *= np.random.uniform(1.3, 2.5)

        return df
    return create_sample_time_series_data,


@app.cell
def __(np, pd, timedelta, datetime):
    def create_financial_time_series():
        """Create financial time series data for analysis."""
        np.random.seed(42)

        # Generate trading days (weekdays only)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        trading_days = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

        n_records = len(trading_days)

        # Simulate stock prices with random walk
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, n_records)  # Daily returns
        prices = [initial_price]

        for i in range(1, n_records):
            price_change = prices[-1] * returns[i]
            new_price = prices[-1] + price_change
            prices.append(max(new_price, 1.0))  # Prevent negative prices

        # Create multiple symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        data = []

        for symbol in symbols:
            # Add symbol-specific variations
            symbol_returns = returns + np.random.normal(0, 0.01, n_records)
            symbol_prices = [initial_price * np.random.uniform(0.8, 1.2)]

            for i in range(1, n_records):
                price_change = symbol_prices[-1] * symbol_returns[i]
                new_price = symbol_prices[-1] + price_change
                symbol_prices.append(max(new_price, 1.0))

            symbol_data = {
                'timestamp': trading_days,
                'symbol': symbol,
                'open': np.array(symbol_prices) * np.random.uniform(0.98, 1.02, n_records),
                'high': np.array(symbol_prices) * np.random.uniform(1.0, 1.05, n_records),
                'low': np.array(symbol_prices) * np.random.uniform(0.95, 1.0, n_records),
                'close': symbol_prices,
                'volume': np.random.lognormal(15, 0.5, n_records).astype(int),
                'market_cap': np.array(symbol_prices) * np.random.uniform(1e9, 5e9, n_records)
            }

            data.append(pd.DataFrame(symbol_data))

        return pd.concat(data, ignore_index=True)
    return create_financial_time_series,


@app.cell
def __(create_financial_time_series, create_sample_time_series_data):
    # Create sample data
    df_ts = create_sample_time_series_data()
    df_financial = create_financial_time_series()

    print(f"Created time series data:")
    print(f" - Operations data: {len(df_ts)} records")
    print(f" - Financial data: {len(df_financial)} records")
    print(f" - Time range: {df_ts['timestamp'].min()} to {df_ts['timestamp'].max()}")

    print("\nSample operations data:")
    print(df_ts.head())

    print("\nSample financial data:")
    print(df_financial.head())
    return df_financial, df_ts


@app.cell
def __(Path, tempfile, df_ts, df_financial, ParquetDataset):
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    print("\n=== Example 1: Time Series Dataset Creation ===")

    # Create partitioned time series dataset
    ds_operations = ParquetDataset.from_pandas(
        df_ts,
        path=temp_path / "operations" / "metrics",
        partition_cols=['year', 'month', 'day_of_week'],
        row_group_size=1000
    )

    # Create financial dataset
    ds_financial = ParquetDataset.from_pandas(
        df_financial,
        path=temp_path / "financial" / "stocks",
        partition_cols=['symbol', 'year', 'month'],
        row_group_size=500
    )

    print(f"\n1.1 Dataset Information:")
    print(f" - Operations dataset: {len(ds_operations.files)} files")
    print(f" - Financial dataset: {len(ds_financial.files)} files")
    print(f" - Operations records: {len(df_ts)}")
    print(f" - Financial records: {len(df_financial)}")
    return ds_financial, ds_operations, temp_dir, temp_path


@app.cell
def __(df_ts, ds_operations):
    # Add time series specific metadata
    ds_operations.metadata.update({
        'description': 'System operations metrics time series',
        'data_frequency': 'hourly',
        'time_range_start': str(df_ts['timestamp'].min()),
        'time_range_end': str(df_ts['timestamp'].max()),
        'metrics': ['metric_cpu', 'metric_memory', 'metric_disk_io', 'metric_network'],
        'granularity': '1 hour',
        'timezone': 'UTC'
    })

    ds_operations.save_metadata()

    print("\n1.2 Time Series Metadata:")
    print("Added time series metadata:")
    for key, value in ds_operations.metadata.items():
        if key in ['description', 'data_frequency', 'granularity', 'timezone']:
            print(f" - {key}: {value}")
    return key, value,


@app.cell
def __(ds_operations):
    # Validate time series integrity
    print("\n1.3 Dataset Validation:")

    table = ds_operations.to_table()
    df_loaded = table.to_pandas()

    # Check for missing timestamps
    expected_hours = (df_loaded['timestamp'].max() - df_loaded['timestamp'].min()).total_seconds() / 3600 + 1
    actual_hours = len(df_loaded)
    completeness = actual_hours / expected_hours * 100

    print(f" - Time completeness: {completeness:.1f}%")
    print(f" - Expected hours: {expected_hours:.0f}")
    print(f" - Actual records: {actual_hours}")

    # Check for duplicates
    duplicates = df_loaded.duplicated('timestamp').sum()
    print(f" - Duplicate timestamps: {duplicates}")

    # Check data quality
    missing_values = df_loaded.isnull().sum()
    print(f" - Missing values: {missing_values.sum()}")
    if missing_values.sum() > 0:
        print("   Columns with missing values:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"     {col}: {missing}")
    return actual_hours, completeness, duplicates, df_loaded, expected_hours, missing_values, table


@app.cell
def __(create_sample_time_series_data, ParquetDataset, temp_path):
    # Create dataset for temporal aggregations
    df = create_sample_time_series_data()
    ds = ParquetDataset.from_pandas(df, path=temp_path / "time_series")
    table = ds.to_table()
    df = table.to_pandas()

    print("\n=== Example 2: Temporal Aggregations ===")

    print(f"\n2.1 Dataset Info:")
    print(f" - Records: {len(df)}")
    print(f" - Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df, ds, table


@app.cell
def __(df):
    # Time-based aggregations
    print("\n2.2 Time-based Aggregations:")

    # Hourly aggregations
    hourly_agg = df.groupby(['timestamp', 'service_name']).agg({
        'metric_cpu': ['mean', 'max', 'min', 'std'],
        'metric_memory': ['mean', 'max'],
        'metric_requests': 'sum',
        'metric_errors': 'sum'
    }).round(2)

    print(f"Hourly aggregation shape: {hourly_agg.shape}")

    # Daily aggregations
    daily_agg = df.groupby(['date', 'service_name']).agg({
        'metric_cpu': ['mean', 'max', 'min'],
        'metric_memory': ['mean', 'max'],
        'metric_requests': 'sum',
        'metric_errors': 'sum',
        'timestamp': 'count'  # record count
    }).round(2)

    print(f"Daily aggregation shape: {daily_agg.shape}")
    print("Sample daily aggregation:")
    print(daily_agg.head())
    return daily_agg, hourly_agg


@app.cell
def __(df):
    # Rolling window calculations
    print("\n2.3 Rolling Window Calculations:")

    # Calculate rolling statistics
    df_sorted = df.sort_values('timestamp')
    df_sorted['cpu_rolling_mean_24h'] = df_sorted['metric_cpu'].rolling(window=24, min_periods=1).mean()
    df_sorted['cpu_rolling_std_24h'] = df_sorted['metric_cpu'].rolling(window=24, min_periods=1).std()
    df_sorted['requests_rolling_sum_7d'] = df_sorted['metric_requests'].rolling(window=24*7, min_periods=1).sum()

    rolling_sample = df_sorted[['timestamp', 'metric_cpu', 'cpu_rolling_mean_24h', 'cpu_rolling_std_24h']].head(48)
    print("24-hour rolling statistics (sample):")
    print(rolling_sample)
    return df_sorted, rolling_sample


@app.cell
def __(df_sorted):
    # Expanding window calculations
    print("\n2.4 Expanding Window Calculations:")

    # Expanding window calculations
    df_sorted['cpu_expanding_mean'] = df_sorted['metric_cpu'].expanding(min_periods=1).mean()
    df_sorted['cpu_expanding_max'] = df_sorted['metric_cpu'].expanding(min_periods=1).max()
    df_sorted['requests_expanding_sum'] = df_sorted['metric_requests'].expanding(min_periods=1).sum()

    print(f"Expanding window calculations added")
    print(f" - Expanding mean CPU: {df_sorted['cpu_expanding_mean'].iloc[-1]:.2f}")
    print(f" - Expanding max CPU: {df_sorted['cpu_expanding_max'].iloc[-1]:.2f}")
    print(f" - Expanding sum requests: {df_sorted['requests_expanding_sum'].iloc[-1]:.0f}")


@app.cell
def __(df):
    # Time-based filtering
    print("\n2.5 Time-based Filtering:")

    # Filter by time ranges
    q1_data = df[df['timestamp'].dt.quarter == 1]
    weekend_data = df[df['is_weekend']]
    business_hours_data = df[df['is_business_hours']]

    print(f"Q1 2023 data: {len(q1_data)} records")
    print(f"Weekend data: {len(weekend_data)} records")
    print(f"Business hours data: {len(business_hours_data)} records")

    # Show distribution of data by time period
    print("\nData distribution by quarter:")
    quarterly_counts = df.groupby('quarter').size()
    for quarter, count in quarterly_counts.items():
        print(f"  Q{quarter}: {count} records")
    return business_hours_data, q1_data, quarterly_counts, weekend_data


@app.cell
def __(create_sample_time_series_data, ParquetDataset, temp_path):
    # Create dataset for seasonal analysis
    df_ts = create_sample_time_series_data()
    ds = ParquetDataset.from_pandas(df_ts, path=temp_path / "seasonal_analysis")
    df = ds.to_table().to_pandas()

    print("\n=== Example 3: Seasonal Analysis ===")

    print(f"\n3.1 Preparing Data for Analysis:")

    # Focus on web-server data for cleaner analysis
    web_server_data = df[df['service_name'] == 'web-server'].copy()
    web_server_data = web_server_data.sort_values('timestamp')
    web_server_data = web_server_data.set_index('timestamp')

    # Resample to daily for clearer seasonality
    daily_data = web_server_data['metric_cpu'].resample('D').mean()

    print(f" - Web server daily data: {len(daily_data)} days")
    return daily_data, df, ds, web_server_data


@app.cell
def __(daily_data, plt):
    # Trend analysis
    print("\n3.2 Trend Analysis:")

    # Calculate moving averages for trend
    ma_7d = daily_data.rolling(window=7, center=True).mean()
    ma_30d = daily_data.rolling(window=30, center=True).mean()

    print("7-day and 30-day moving averages calculated")

    # Plot trend analysis
    plt.figure(figsize=(12, 6))
    plt.plot(daily_data.index, daily_data, label='Daily CPU', alpha=0.7)
    plt.plot(ma_7d.index, ma_7d, label='7-day MA', linewidth=2)
    plt.plot(ma_30d.index, ma_30d, label='30-day MA', linewidth=2)
    plt.title('CPU Usage Trend Analysis')
    plt.xlabel('Date')
    plt.ylabel('CPU Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return ma_30d, ma_7d


@app.cell
def __(daily_data, plt, seasonal_decompose):
    # Seasonal decomposition
    print("\n3.3 Seasonal Decomposition:")

    # Perform seasonal decomposition (if enough data points)
    if len(daily_data) >= 14:  # Minimum for seasonal decomposition
        try:
            # Use a simple period of 7 for weekly seasonality
            decomposition = seasonal_decompose(daily_data.dropna(), model='additive', period=7)

            print("Seasonal decomposition completed:")
            print(f" - Trend component: {len(decomposition.trend.dropna())} points")
            print(f" - Seasonal component: {len(decomposition.seasonal.dropna())} points")
            print(f" - Residual component: {len(decomposition.resid.dropna())} points")

            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            decomposition.observed.plot(ax=axes[0], title='Observed')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Seasonal decomposition failed: {e}")
    return decomposition, e, fig, axes


@app.cell
def __(df, plt, sns):
    # Weekly patterns
    print("\n3.4 Weekly Patterns:")

    # Analyze weekly patterns
    weekly_patterns = df.groupby(['day_of_week', 'hour'])['metric_cpu'].agg(['mean', 'std']).round(2)

    print("Average CPU usage by day and hour:")
    print(weekly_patterns.head(10))

    # Create heatmap of weekly patterns
    weekly_heatmap = weekly_patterns['mean'].unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(weekly_heatmap, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Average CPU Usage'})
    plt.title('Weekly CPU Usage Patterns')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week (0=Monday)')
    plt.tight_layout()
    plt.show()
    return weekly_heatmap, weekly_patterns


@app.cell
def __(df, plt):
    # Monthly patterns
    print("\n3.5 Monthly Patterns:")

    # Monthly analysis
    monthly_stats = df.groupby('month')['metric_cpu'].agg(['mean', 'std', 'min', 'max']).round(2)
    monthly_stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print("Monthly CPU usage patterns:")
    print(monthly_stats)

    # Plot monthly patterns
    plt.figure(figsize=(10, 6))
    monthly_stats['mean'].plot(kind='bar', yerr=monthly_stats['std'], alpha=0.7)
    plt.title('Monthly CPU Usage Patterns')
    plt.xlabel('Month')
    plt.ylabel('Average CPU Usage')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return monthly_stats,


@app.cell
def __(plt, stats, web_server_data):
    # Anomaly detection
    print("\n3.6 Anomaly Detection:")

    # Simple anomaly detection using Z-score
    web_server_data['cpu_zscore'] = stats.zscore(web_server_data['metric_cpu'])
    anomalies = web_server_data[abs(web_server_data['cpu_zscore']) > 3]

    print(f"Detected {len(anomalies)} anomalies (Z-score > 3)")
    if len(anomalies) > 0:
        print("Top 5 anomalies:")
        print(anomalies[['metric_cpu', 'cpu_zscore']].head())

        # Plot anomalies
        plt.figure(figsize=(12, 6))
        plt.plot(web_server_data.index, web_server_data['metric_cpu'], label='CPU Usage', alpha=0.7)
        plt.scatter(anomalies.index, anomalies['metric_cpu'], color='red', label='Anomalies', s=50, zorder=5)
        plt.title('CPU Usage with Anomalies')
        plt.xlabel('Timestamp')
        plt.ylabel('CPU Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return anomalies,


@app.cell
def __(create_sample_time_series_data, ParquetDataset, temp_path):
    # Create dataset for forecasting
    df_ts = create_sample_time_series_data()
    ds = ParquetDataset.from_pandas(df_ts, path=temp_path / "forecasting")
    df = ds.to_table().to_pandas()

    print("\n=== Example 4: Time Series Forecasting ===")

    print(f"\n4.1 Preparing Data for Forecasting:")

    # Use web-server data for forecasting
    web_data = df[df['service_name'] == 'web-server'].copy()
    web_data = web_data.sort_values('timestamp')

    # Create daily aggregation for forecasting
    daily_cpu = web_data.groupby(web_data['timestamp'].dt.date)['metric_cpu'].mean()

    print(f"Daily CPU data points: {len(daily_cpu)}")

    # Plot the time series
    plt.figure(figsize=(12, 6))
    daily_cpu.plot()
    plt.title('Daily CPU Usage - Web Server')
    plt.xlabel('Date')
    plt.ylabel('CPU Usage')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return daily_cpu, df, ds, web_data


@app.cell
def __(daily_cpu, plt):
    # Simple moving average forecast
    print("\n4.2 Simple Moving Average Forecast:")

    # Simple moving average forecast
    window_sizes = [7, 14, 30]
    forecasts = {}

    for window in window_sizes:
        ma_forecast = daily_cpu.rolling(window=window).mean()
        forecasts[f'MA_{window}'] = ma_forecast
        print(f"MA({window}) forecast calculated")

    # Plot moving averages
    plt.figure(figsize=(12, 6))
    daily_cpu.plot(label='Actual', alpha=0.7)
    for window in window_sizes:
        forecasts[f'MA_{window}'].plot(label=f'MA({window})')
    plt.title('Moving Average Forecasts')
    plt.xlabel('Date')
    plt.ylabel('CPU Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return forecasts, window_sizes, window


@app.cell
def __(daily_cpu, plt):
    # Exponential smoothing
    print("\n4.3 Exponential Smoothing:")

    # Exponential smoothing
    alpha_values = [0.1, 0.3, 0.5]
    for alpha in alpha_values:
        exp_smooth = daily_cpu.ewm(alpha=alpha, adjust=False).mean()
        forecasts[f'ExpSmooth_{alpha}'] = exp_smooth
        print(f"Exponential smoothing (α={alpha}) calculated")

    # Plot exponential smoothing
    plt.figure(figsize=(12, 6))
    daily_cpu.plot(label='Actual', alpha=0.7)
    for alpha in alpha_values:
        forecasts[f'ExpSmooth_{alpha}'].plot(label=f'Exp Smooth (α={alpha})')
    plt.title('Exponential Smoothing Forecasts')
    plt.xlabel('Date')
    plt.ylabel('CPU Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return alpha, alpha_values, exp_smooth


@app.cell
def __(daily_cpu, forecasts, np):
    # Forecast accuracy metrics
    print("\n4.4 Forecast Accuracy Metrics:")

    # Calculate forecast accuracy (using last 30 days as test set)
    train_size = len(daily_cpu) - 30
    train_data = daily_cpu[:train_size]
    test_data = daily_cpu[train_size:]

    print(f"Training set: {len(train_data)} points")
    print(f"Test set: {len(test_data)} points")

    # Calculate MAE for different methods
    mae_results = {}
    for method, forecast in forecasts.items():
        if len(forecast) > train_size:
            forecast_test = forecast[train_size:]
            if len(forecast_test) == len(test_data):
                mae = np.mean(abs(test_data - forecast_test))
                mae_results[method] = mae

    print("\nForecast accuracy (MAE) on test set:")
    for method, mae in sorted(mae_results.items(), key=lambda x: x[1]):
        print(f" - {method}: {mae:.2f}")
    return forecast_test, mae, mae_results, method, test_data, train_data, train_size


@app.cell
def __(daily_cpu, np, plt, stats):
    # Trend analysis
    print("\n4.5 Trend Analysis:")

    # Linear trend analysis
    time_idx = np.arange(len(daily_cpu))
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, daily_cpu)

    print(f"Trend analysis results:")
    print(f" - Slope (trend): {slope:.4f} per day")
    print(f" - R-squared: {r_value**2:.4f}")
    print(f" - P-value: {p_value:.4f}")

    if p_value < 0.05:
        trend_direction = "increasing" if slope > 0 else "decreasing"
        print(f" - Significant {trend_direction} trend detected")

    # Plot trend line
    plt.figure(figsize=(12, 6))
    daily_cpu.plot(label='Actual', alpha=0.7)
    trend_line = intercept + slope * time_idx
    plt.plot(daily_cpu.index, trend_line, label='Trend Line', linewidth=2, color='red')
    plt.title('Trend Analysis')
    plt.xlabel('Date')
    plt.ylabel('CPU Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return intercept, p_value, r_value, slope, std_err, time_idx, time_idx, trend_direction, trend_line


@app.cell
def __(daily_cpu, np, seasonal_decompose):
    # Seasonal strength
    print("\n4.6 Seasonal Strength:")

    # Calculate seasonal strength
    if len(daily_cpu) >= 14:
        try:
            # Simple seasonal strength calculation
            decomp = seasonal_decompose(daily_cpu.dropna(), model='additive', period=7)
            seasonal_strength = np.var(decomp.seasonal.dropna()) / np.var(daily_cpu.dropna())
            print(f"Seasonal strength: {seasonal_strength:.4f}")

            if seasonal_strength > 0.1:
                print("Strong seasonal patterns detected")
            elif seasonal_strength > 0.05:
                print("Moderate seasonal patterns detected")
            else:
                print("Weak seasonal patterns detected")

        except Exception as e:
            print(f"Seasonal strength calculation failed: {e}")
    return decomp, e, seasonal_strength


@app.cell
def __(create_sample_time_series_data, pd):
    # Create large time series dataset
    df_ts = create_sample_time_series_data()
    # Duplicate to make it larger for performance testing
    df_large = pd.concat([df_ts] * 10, ignore_index=True)

    print("\n=== Example 5: Performance Optimization ===")

    print(f"\n5.1 Large Dataset Created:")
    print(f" - Records: {len(df_large)}")
    print(f" - Memory usage: {df_large.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    return df_large, df_ts


@app.cell
def __(ParquetDataset, temp_path, time, df_large):
    # Test different partitioning strategies
    print("\n5.2 Partitioning Strategies:")

    strategies = [
        ("by_year_month", ['year', 'month']),
        ("by_year_month_day", ['year', 'month', 'date']),
        ("by_service_time", ['service_name', 'year', 'month']),
        ("by_region_time", ['region', 'year', 'month'])
    ]

    strategy_results = []

    for strategy_name, partition_cols in strategies:
        start_time = time.time()

        ds = ParquetDataset.from_pandas(
            df_large,
            path=temp_path / f"strategy_{strategy_name}",
            partition_cols=partition_cols,
            row_group_size=2000
        )

        end_time = time.time()
        creation_time = end_time - start_time

        # Test query performance
        table = ds.to_table()
        start_query = time.time()
        result = table.filter((table.year == 2023) & (table.month == 6))
        end_query = time.time()
        query_time = end_query - start_query

        strategy_results.append({
            'strategy': strategy_name,
            'creation_time': creation_time,
            'query_time': query_time,
            'files': len(ds.files),
            'partitions': len(partition_cols)
        })

        print(f" - {strategy_name}: {creation_time:.3f}s creation, {query_time:.3f}s query, {len(ds.files)} files")
    return creation_time, ds, end_query, end_time, partition_cols, result, start_query, start_time, strategy_name, strategy_results, strategies, table


@app.cell
def __(strategy_results):
    # Strategy comparison
    print("\n5.3 Strategy Comparison:")
    print("Strategy | Creation Time | Query Time | Files | Partitions")
    print("-" * 65)
    for result in strategy_results:
        print(f"{result['strategy']:<15} | {result['creation_time']:<13.3f} | {result['query_time']:<10.3f} | {result['files']:<5} | {result['partitions']:<10}")

    # Find best strategy
    best_strategy = min(strategy_results, key=lambda x: x['query_time'])
    print(f"\nBest strategy for queries: {best_strategy['strategy']}")
    return best_strategy, result


@app.cell
def __(ParquetDataset, temp_path, df_large):
    # Create optimal dataset
    print("\n5.4 Optimal Dataset Configuration:")

    # Create optimal dataset for time series
    ds_optimized = ParquetDataset.from_pandas(
        df_large,
        path=temp_path / "optimized_timeseries",
        partition_cols=['year', 'month', 'service_name'],
        row_group_size=1440  # 1 day of hourly data
    )

    print(f"Optimized dataset created:")
    print(f" - Row group size: 1440 (1 day)")
    print(f" - Partitions: year, month, service_name")
    print(f" - Files: {len(ds_optimized.files)}")
    return ds_optimized,


@app.cell
def __(ds_optimized, time):
    # Query performance tests
    print("\n5.5 Query Performance Tests:")

    table = ds_optimized.to_table()

    # Test different query patterns
    queries = [
        ("Time range filter", lambda t: t.filter((t.month >= 6) & (t.month <= 8))),
        ("Service filter", lambda t: t.filter(t.service_name == 'web-server')),
        ("Complex filter", lambda t: t.filter((t.month == 7) & (t.service_name == 'database') & (t.metric_cpu > 200))),
        ("Aggregation query", lambda t: t.to_pandas().groupby(['month', 'service_name'])['metric_cpu'].mean())
    ]

    for query_name, query_func in queries:
        start_time = time.time()
        result = query_func(table)
        end_time = time.time()

        result_size = len(result) if hasattr(result, '__len__') else 'N/A'
        print(f" - {query_name}: {end_time - start_time:.3f}s, result: {result_size}")
    return end_time, queries, query_func, query_name, result, result_size, start_time, table


@app.cell
def __(ds_optimized, time):
    # Memory-efficient processing
    print("\n5.6 Memory-Efficient Processing:")

    # Test memory-efficient processing
    scanner = ds_optimized.to_arrow_scanner(
        columns=['timestamp', 'metric_cpu', 'service_name'],
        filter=(ds_optimized.to_table().year == 2023) & (ds_optimized.to_table().month == 7)
    )

    batch_count = 0
    total_rows = 0
    start_time = time.time()

    for batch in scanner.to_batches():
        batch_count += 1
        total_rows += batch.num_rows
        if batch_count >= 100:  # Limit for demo
            break

    end_time = time.time()
    print(f"Batch processing: {end_time - start_time:.3f}s")
    print(f"Processed {batch_count} batches, {total_rows} rows")
    return batch, batch_count, end_time, scanner, start_time, total_rows


@app.cell
def __(shutil, temp_dir):
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")

    print("\n" + "="*50)
    print("Time series data examples completed!")
    print("\nKey Takeaways:")
    print("- Time series dataset creation and management")
    print("- Temporal aggregations and windowing operations")
    print("- Seasonal analysis and trend detection")
    print("- Time series forecasting concepts")
    print("- Performance optimization for time-based data")
    return shutil,


if __name__ == "__main__":
    app.run()