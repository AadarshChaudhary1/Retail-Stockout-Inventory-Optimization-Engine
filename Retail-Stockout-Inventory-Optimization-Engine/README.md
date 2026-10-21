# Retail Stockout & Inventory Optimization Engine

A runnable Python project that analyzes retail sales and current inventory, forecasts near-term demand, calculates reorder recommendations, scores inventory risk, and exports an Excel workbook.

## Features
- Generates a reproducible sample retail dataset if input files are missing.
- Uses Pandas and NumPy for data preparation and analysis.
- Uses SQLite SQL queries for product-level summaries.
- Trains a Random Forest regression model to forecast demand over the next 7 days.
- Calculates safety stock, reorder point, and suggested order quantity.
- Classifies products as Critical, At Risk, Healthy, or Overstocked.
- Exports an Excel workbook with KPI summary, inventory recommendations, and daily sales history.

## Setup
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The program writes:
- `data/sales_history.csv`
- `data/inventory.csv`
- `output/inventory_optimization_report.xlsx`
- `output/sql_product_summary.csv`

## Input files
If you replace the generated CSV files, keep these columns:

**sales_history.csv**
- `date`, `product_id`, `units_sold`

**inventory.csv**
- `product_id`, `product_name`, `category`, `current_stock`, `supplier_lead_time_days`, `unit_cost`, `target_cover_days`

The sample data is synthetic and intended for demonstration, not real purchasing decisions.

## Important assumptions
- Forecast horizon is 7 days.
- Forecasting is performed per product using recent daily sales, weekday, and day-of-year features.
- Safety stock is estimated as `1.65 × standard deviation of daily demand × sqrt(lead time)`.
- Reorder point is expected lead-time demand plus safety stock.
- Suggested order quantity targets forecast demand during lead time plus target cover days, less current stock, floored at zero.
- Risk labels are rule-based; tune thresholds and service levels for a real business.
