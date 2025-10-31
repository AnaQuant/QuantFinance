"""
Data Fetching Utilities for Fixed Income Data.

This module provides functions to fetch yield curve data from various sources,
including US Treasury and other government bond data.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
from .yield_curve import YieldCurve


def fetch_us_treasury_data(
    year: Optional[int] = None,
    url: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch US Treasury yield curve data from treasury.gov.

    Parameters
    ----------
    year : int, optional
        Year to fetch data for (default: current year)
    url : str, optional
        Custom URL to fetch from (overrides year parameter)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Date', '1 Mo', '2 Mo', '3 Mo', '4 Mo',
        '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']

    Examples
    --------
    >>> df = fetch_us_treasury_data(year=2025)
    >>> df.head()
    """
    if url is None:
        if year is None:
            year = datetime.now().year
        url = (
            f"https://home.treasury.gov/resource-center/data-chart-center/"
            f"interest-rates/TextView?type=daily_treasury_yield_curve&"
            f"field_tdr_date_value={year}"
        )

    # Read HTML tables from the URL
    data_dict = pd.read_html(url)
    df = data_dict[0]  # First table contains the yield curve data

    # Select relevant columns
    columns_to_keep = [
        'Date', '1 Mo', '2 Mo', '3 Mo', '4 Mo', '6 Mo',
        '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr'
    ]

    # Filter to only existing columns
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()

    # Remove rows with missing data
    df.dropna(inplace=True)

    return df


def parse_treasury_data_to_yield_curves(
    df: pd.DataFrame,
    date_column: str = 'Date',
    date_format: Optional[str] = None
) -> Dict[datetime, YieldCurve]:
    """
    Parse Treasury DataFrame into a dictionary of YieldCurve objects.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from fetch_us_treasury_data()
    date_column : str
        Name of the date column
    date_format : str, optional
        Date format string (e.g., '%m/%d/%Y')

    Returns
    -------
    dict
        Dictionary mapping datetime objects to YieldCurve objects

    Examples
    --------
    >>> df = fetch_us_treasury_data(year=2025)
    >>> curves = parse_treasury_data_to_yield_curves(df)
    >>> latest_curve = curves[max(curves.keys())]
    """
    # Mapping of column names to tenors in years
    tenor_mapping = {
        '1 Mo': 1/12,
        '2 Mo': 2/12,
        '3 Mo': 3/12,
        '4 Mo': 4/12,
        '6 Mo': 6/12,
        '1 Yr': 1.0,
        '2 Yr': 2.0,
        '3 Yr': 3.0,
        '5 Yr': 5.0,
        '7 Yr': 7.0,
        '10 Yr': 10.0,
        '20 Yr': 20.0,
        '30 Yr': 30.0
    }

    # Parse dates
    if date_format is not None:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    else:
        df[date_column] = pd.to_datetime(df[date_column])

    # Create yield curves for each date
    yield_curves = {}

    for idx, row in df.iterrows():
        date = row[date_column]
        tenors = []
        yields = []

        for col, tenor in tenor_mapping.items():
            if col in df.columns and pd.notna(row[col]):
                tenors.append(tenor)
                # Convert from percentage to decimal
                yields.append(row[col] / 100.0)

        if len(tenors) > 0:
            yield_curves[date] = YieldCurve(
                tenors=tenors,
                yields=yields,
                date=date,
                currency='USD'
            )

    return yield_curves


def load_yield_curve_from_excel(
    file_path: str,
    sheet_name: Optional[str] = None,
    index_col: int = 0,
    parse_dates: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load yield curve data from an Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel file
    sheet_name : str, optional
        Specific sheet name to load (default: load all sheets)
    index_col : int
        Column to use as row labels (default: 0)
    parse_dates : bool
        Whether to parse dates in the index (default: True)

    Returns
    -------
    dict
        Dictionary mapping sheet names to DataFrames

    Examples
    --------
    >>> data = load_yield_curve_from_excel('yield_curves.xlsx')
    >>> usd_data = data['USD_Z0']
    """
    try:
        if sheet_name is not None:
            yield_data = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                index_col=index_col,
                parse_dates=parse_dates
            )
            # Return as dict for consistency
            return {sheet_name: yield_data}
        else:
            # Load all sheets
            sheet_names = pd.ExcelFile(file_path).sheet_names
            yield_data = pd.read_excel(
                file_path,
                sheet_name=sheet_names,
                index_col=index_col,
                parse_dates=parse_dates
            )
            return yield_data
    except Exception as e:
        raise RuntimeError(f"Error reading file '{file_path}': {e}")


def create_yield_curve_from_excel_row(
    df: pd.DataFrame,
    row_index: int = -1,
    currency: Optional[str] = None
) -> YieldCurve:
    """
    Create a YieldCurve object from a row in an Excel DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with tenors as columns and dates as index
    row_index : int
        Row index to extract (default: -1 for latest)
    currency : str, optional
        Currency code

    Returns
    -------
    YieldCurve
        YieldCurve object for the specified date

    Examples
    --------
    >>> data = load_yield_curve_from_excel('yield_curves.xlsx')
    >>> eur_df = data['EUR_Z0']
    >>> latest_curve = create_yield_curve_from_excel_row(eur_df, currency='EUR')
    """
    # Get the row
    row = df.iloc[row_index]
    date = df.index[row_index] if hasattr(df.index, '__getitem__') else None

    # Parse tenors from column names (assuming they are floats)
    tenors = []
    yields = []

    for col in df.columns:
        try:
            tenor = float(col)
            if pd.notna(row[col]):
                tenors.append(tenor)
                # Assume yields are in percentage form
                yields.append(row[col] / 100.0 if row[col] > 1 else row[col])
        except (ValueError, TypeError):
            continue

    return YieldCurve(
        tenors=tenors,
        yields=yields,
        date=pd.to_datetime(date) if date is not None else None,
        currency=currency
    )


def fetch_historical_treasury_data(
    start_year: int,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch historical US Treasury data for multiple years.

    Parameters
    ----------
    start_year : int
        Starting year
    end_year : int, optional
        Ending year (default: current year)

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with data from all years

    Examples
    --------
    >>> df = fetch_historical_treasury_data(2020, 2025)
    """
    if end_year is None:
        end_year = datetime.now().year

    dfs = []
    for year in range(start_year, end_year + 1):
        try:
            df = fetch_us_treasury_data(year=year)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to fetch data for year {year}: {e}")

    if len(dfs) == 0:
        raise ValueError("No data could be fetched for the specified years")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates and sort by date
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.drop_duplicates(subset=['Date'])
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)

    return combined_df
