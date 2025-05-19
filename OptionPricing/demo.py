#!/usr/bin/env python3
"""
Options Pricing Demo
A working example to test the options pricing framework with real data
"""

import datetime as dt
from option_pricing_framework import YahooFinanceProvider, OptionsPricer, OptionContract

def main():
    print("========== Options Pricing Demo ==========")
    print("This demo will fetch real market data and price options")
    
    # Create data provider and pricer
    data_provider = YahooFinanceProvider()
    pricer = OptionsPricer(data_provider)
    
    # Ask user for ticker
    ticker = input("\nEnter ticker symbol (default: SPY): ").strip().upper() or "SPY"
    
    # Get market data
    print(f"\nFetching market data for {ticker}...")
    try:
        market_data = pricer.get_market_data(ticker)
        spot_price = market_data['current_price']
        hist_vol = market_data['historical_volatility']
        print(f"Current price: ${spot_price:.2f}")
        print(f"Historical volatility: {hist_vol:.2%}")
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return
    
    # Get available option expiry dates
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        expiry_dates = stock.options
        
        if not expiry_dates:
            print(f"No options data available for {ticker}")
            return
        
        print("\nAvailable expiry dates:")
        for i, date in enumerate(expiry_dates[:10]):  # Show first 10 dates
            print(f"  [{i+1}] {date}")
        
        # Ask user to select expiry
        selection = input("\nSelect expiry date (number) or enter 'all' to show all: ").strip()
        
        if selection.lower() == 'all':
            print("\nAll available expiry dates:")
            for i, date in enumerate(expiry_dates):
                print(f"  [{i+1}] {date}")
            selection = input("\nSelect expiry date (number): ").strip()
        
        try:
            expiry_idx = int(selection) - 1
            if expiry_idx < 0 or expiry_idx >= len(expiry_dates):
                print("Invalid selection, using first expiry date")
                expiry_idx = 0
        except:
            print("Invalid selection, using first expiry date")
            expiry_idx = 0
        
        selected_expiry = expiry_dates[expiry_idx]
        print(f"\nSelected expiry date: {selected_expiry}")
        expiry_dt = dt.datetime.strptime(selected_expiry, '%Y-%m-%d')
        
        # Get options chain
        print(f"\nFetching options chain for {ticker} expiring on {selected_expiry}...")
        options_chain = pricer.get_options_data(ticker, expiry_dt)
        
        if options_chain.empty:
            print("No options data available")
            return
        
        # Allow user to choose call or put
        option_type = input("\nSelect option type (call/put) [default: call]: ").strip().lower() or "call"
        if option_type not in ["call", "put"]:
            print("Invalid option type, using call")
            option_type = "call"
        
        # Filter options by type
        type_options = options_chain[options_chain['optionType'] == option_type]
        type_options = type_options.sort_values(by='strike')
        
        if type_options.empty:
            print(f"No {option_type} options available")
            return
        
        # Show strikes around the money
        near_money = type_options[
            (type_options['strike'] >= spot_price * 0.8) & 
            (type_options['strike'] <= spot_price * 1.2)
        ]
        
        if near_money.empty:
            near_money = type_options  # Use all if no near-money options
        
        print(f"\nAvailable {option_type} options around the money:")
        print(f"Current price: ${spot_price:.2f}")
        
        # Display options in a table format
        print("\n{:<5} {:<10} {:<10} {:<10} {:<10}".format(
            "No.", "Strike", "Last", "Volume", "Open Int"
        ))
        print("-" * 50)
        
        for i, (_, row) in enumerate(near_money.iterrows()):
            print("{:<5} {:<10.2f} {:<10.2f} {:<10} {:<10}".format(
                i+1,
                row['strike'],
                row['lastPrice'],
                int(row['volume']) if not pd.isna(row['volume']) else 0,
                int(row['openInterest']) if not pd.isna(row['openInterest']) else 0
            ))
        
        # Ask user to select a strike
        strike_selection = input("\nSelect strike (number): ").strip()
        
        try:
            strike_idx = int(strike_selection) - 1
            if strike_idx < 0 or strike_idx >= len(near_money):
                print("Invalid selection, using ATM option")
                # Find closest to ATM
                near_money['diff'] = abs(near_money['strike'] - spot_price)
                strike_idx = near_money['diff'].idxmin()
                selected_option = near_money.loc[strike_idx]
            else:
                selected_option = near_money.iloc[strike_idx]
        except:
            print("Invalid selection, using ATM option")
            # Find closest to ATM
            near_money['diff'] = abs(near_money['strike'] - spot_price)
            strike_idx = near_money['diff'].idxmin()
            selected_option = near_money.loc[strike_idx]
        
        strike = selected_option['strike']
        market_price = selected_option['lastPrice']
        
        print(f"\nSelected {ticker} {strike:.2f} {option_type.upper()} expiring on {selected_expiry}")
        print(f"Market price: ${market_price:.2f}")
        
        # Create option contract
        option = OptionContract(
            ticker=ticker,
            strike=strike,
            expiry_date=expiry_dt,
            option_type=option_type,
            market_price=market_price
        )
        
        # Compare pricing models
        print("\nComparing pricing models...")
        results = pricer.compare_models(option)
        
        # Print comparison table
        print("\n{:<15} {:<10} {:<15} {:<10}".format(
            "Model", "Price", "Implied Vol", "Difference"
        ))
        print("-" * 55)
        
        for model_name, result in results.items():
            price = result['price']
            vol = result['inputs']['volatility']
            diff = result['price_diff'] if result['price_diff'] is not None else float('nan')
            
            print("{:<15} ${:<9.2f} {:<15.2%} ${:<9.2f}".format(
                model_name.replace('_', ' ').title(),
                price,
                vol,
                diff
            ))
        
        # Print Greeks for Black-Scholes model
        print("\nBlack-Scholes Greeks:")
        bs_greeks = results['black_scholes']['greeks']
        for greek, value in bs_greeks.items():
            print(f"  {greek.capitalize()}: {value:.6f}")
        
        # Offer to run volatility surface plot
        plot_choice = input("\nGenerate volatility surface plot? (y/n) [default: n]: ").strip().lower()
        if plot_choice == 'y':
            print("\nGenerating volatility surface (this may take a moment)...")
            try:
                import matplotlib.pyplot as plt
                fig = pricer.plot_volatility_surface(ticker)
                if fig:
                    plt.show()
                else:
                    print("Could not generate volatility surface - insufficient data")
            except Exception as e:
                print(f"Error generating plot: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import pandas as pd  # Import here for near_money operations
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("\nDemo complete")