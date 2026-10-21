Retail Stockout & Inventory Optimization Engine
A Python-based retail analytics project that identifies potential stockouts and overstocking, forecasts short-term product demand, and recommends replenishment quantities. It combines data analysis, SQL aggregation, machine-learning forecasting, inventory rules, and an Excel report.
Note: The included dataset is synthetic and intended for demonstration and learning. Forecasts and recommendations have not been validated for production purchasing decisions.

Features
- Retail data preparation: validates and cleans sales and inventory CSV files using Pandas and NumPy.
- Demand forecasting: trains a Random Forest regression model to estimate daily demand over the next seven days.
- SQL analysis: aggregates product sales and inventory value using SQLite.
- Inventory planning: calculates estimated lead-time demand, safety stock, reorder points, target stock levels, and suggested order quantities.
- Risk classification: labels products as Critical, At Risk, Healthy, or Overstocked.
- Excel reporting: exports KPI summaries, product recommendations, SQL summaries, sales history, and a risk-category chart.
Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn (RandomForestRegressor)
- SQLite / SQL
- Excel reporting with OpenPyXL
Project Structure
Retail-Stockout-Inventory-Optimization-Engine/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── inventory.csv
│   └── sales_history.csv
├── sql/
│   └── analysis_queries.sql
└── output/
    ├── inventory_optimization_report.xlsx
    └── sql_product_summary.csv
The program creates sample CSV inputs automatically if either required input file is missing. Generated reports are written to output/.
Getting Started
1. Requirements
- Python 3.10 or newer recommended
- pip
2. Clone the repository
git clone https://github.com/AadarshChaudhary1/Retail-Stockout-Inventory-Optimization-Engine.git
cd Retail-Stockout-Inventory-Optimization-Engine
3. Create and activate a virtual environment (recommended)
Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
4. Install dependencies
pip install -r requirements.txt
5. Run the project
python main.py
When execution completes, check:
- output/inventory_optimization_report.xlsx
- output/sql_product_summary.csv
The console also prints a product-level inventory recommendation table.
Input Data
data/sales_history.csv
Column	Description
date	Date of the sales record
product_id	Product identifier
units_sold	Number of units sold


data/inventory.csv
Column	Description
product_id	Unique product identifier
product_name	Product name
category	Product category
current_stock	Current units available
supplier_lead_time_days	Estimated supplier delivery lead time
unit_cost	Cost per unit
target_cover_days	Desired days of stock coverage after replenishment


If using your own data, preserve these column names and ensure product_id values match across both files.
How the Inventory Logic Works
1. Forecast demand: estimates daily demand for each product over the next seven days.
2. Estimate lead-time demand: multiplies forecast daily demand by supplier lead time.
3. Calculate safety stock: uses historical demand variability and a configurable service-level factor.
4. Calculate reorder point: combines estimated lead-time demand and safety stock.
5. Recommend order quantity: estimates the target stock level from lead time, target cover days, and safety stock, then subtracts current stock.
6. Assign risk status: compares current inventory with demand and target-stock thresholds to suggest an action.
The safety-stock factor and risk thresholds are configurable in main.py. They are simplified demonstration assumptions and should be calibrated to actual service-level goals, forecast error, supplier reliability, and business constraints before operational use.
SQL Analysis
Example queries are available in sql/analysis_queries.sql. They cover:
- Total units sold by product
- Average daily demand
- Inventory value by category
- Products whose stock is below estimated demand during supplier lead time
The application also runs a SQLite aggregation and exports the result to output/sql_product_summary.csv.
Output Workbook
The Excel workbook contains:
- Dashboard: summary KPIs and inventory risk counts
- Recommendations: forecast, safety stock, reorder point, order quantity, risk label, and recommended action
- SQL Summary: product-level sales and inventory metrics
- Sales History: input sales records
Limitations and Future Improvements
- The supplied dataset is synthetic, not real retail data.
- Forecast quality should be evaluated on a chronological holdout set using metrics such as MAE or RMSE.
- The current forecasting approach is a baseline; promotions, holidays, stockout-censored demand, and external factors are not modeled.
- The inventory policy uses simplified assumptions and does not account for minimum order quantities, pack sizes, budgets, or multiple suppliers.
- Potential improvements include forecast backtesting, seasonality-aware models, automated data ingestion, a dashboard UI, and configurable business rules.
License
No license is specified yet. Add a LICENSE file if you want to clearly define how others may use, modify, and distribute this project.
