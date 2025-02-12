import csv
from collections import defaultdict

class PortfolioManager:
    def __init__(self):
        self.portfolios = {}  # {portfolio_name: {component_name: shares}}
        self.stock_prices = {}  # {stock_name: price}
        self.dependency_graph = defaultdict(set)  # {component_name: set(portfolios_affected)}
        self.portfolio_cache = {}  # {portfolio_name: cached_price}

    def load_portfolios(self, filename):
        """Loads portfolios from a CSV file."""
        with open(filename, mode='r') as file:
            csv_reader = csv.DictReader(file)
            current_portfolio = None

            for row in csv_reader:
                name, shares = row['NAME'], row['SHARES']
                if shares == '':
                    current_portfolio = name
                    self.portfolios[current_portfolio] = {}
                else:
                    self.portfolios[current_portfolio][name] = int(shares)
                    self.dependency_graph[name].add(current_portfolio)

    def calculate_portfolio_value(self, portfolio_name):
        """Recursively calculates the value of a portfolio."""
        if portfolio_name in self.portfolio_cache:
            return self.portfolio_cache[portfolio_name]

        total_value = 0.0
        for component, shares in self.portfolios[portfolio_name].items():
            if component in self.stock_prices:
                total_value += shares * self.stock_prices[component]
            elif component in self.portfolios:
                sub_value = self.calculate_portfolio_value(component)
                if sub_value is None:
                    return None  # Still missing prices
                total_value += shares * sub_value
            else:
                return None  # Unknown component

        self.portfolio_cache[portfolio_name] = total_value  # Cache result
        return total_value

    def update_prices(self, input_filename, output_filename):
        """Reads stock prices and updates affected portfolios."""
        with open(input_filename, mode='r') as input_file, open(output_filename, mode='w', newline='') as output_file:
            csv_reader = csv.DictReader(input_file)
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(['NAME', 'PRICE'])

            for row in csv_reader:
                stock_name, price = row['NAME'], float(row['PRICE'])
                self.stock_prices[stock_name] = price
                self.portfolio_cache.clear()  # Invalidate cache
                csv_writer.writerow([stock_name, price])

                # Find affected portfolios
                affected = self.get_affected_portfolios(stock_name)
                for portfolio in affected:
                    value = self.calculate_portfolio_value(portfolio)
                    if value is not None:
                        csv_writer.writerow([portfolio, value])


    def get_affected_portfolios(self, name):
        """Finds all portfolios affected by a change in a stock or sub-portfolio."""
        affected = set()  # Set to store affected portfolios
        stack = [name]  # Stack for DFS traversal (Depth first search)

        while stack:
            current = stack.pop()  # Get the next element to process
            for dependent in self.dependency_graph[current]:  # Get portfolios dependent on current
                if dependent not in affected:  # Avoid duplicates
                    affected.add(dependent)  # Mark as affected
                    stack.append(dependent)  # Add to stack for further processing

        return affected



if __name__ == '__main__':
    manager = PortfolioManager()
    manager.load_portfolios('portfolios.csv')
    manager.update_prices('prices.csv', 'portfolio_prices.csv')

