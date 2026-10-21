-- These queries are used for product-level retail inventory analysis.
-- Run them against a database containing:
-- inventory(product_id, product_name, category, current_stock, supplier_lead_time_days, unit_cost, target_cover_days)
-- sales(date, product_id, units_sold)

-- 1. Top products by total units sold
SELECT product_id, SUM(units_sold) AS total_units_sold
FROM sales
GROUP BY product_id
ORDER BY total_units_sold DESC;

-- 2. Average daily demand per product
SELECT product_id, AVG(units_sold) AS avg_daily_demand
FROM sales
GROUP BY product_id
ORDER BY avg_daily_demand DESC;

-- 3. Current inventory value by category
SELECT category, SUM(current_stock * unit_cost) AS inventory_value
FROM inventory
GROUP BY category
ORDER BY inventory_value DESC;

-- 4. Products with stock below average demand during supplier lead time
WITH demand AS (
    SELECT product_id, AVG(units_sold) AS avg_daily_demand
    FROM sales
    GROUP BY product_id
)
SELECT i.product_id, i.product_name, i.current_stock,
       d.avg_daily_demand, i.supplier_lead_time_days
FROM inventory i
JOIN demand d ON i.product_id = d.product_id
WHERE i.current_stock < d.avg_daily_demand * i.supplier_lead_time_days
ORDER BY i.current_stock ASC;
