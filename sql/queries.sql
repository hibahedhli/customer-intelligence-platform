-- name: monthly_revenue
SELECT substr(date, 1, 7) AS month,
       ROUND(SUM(transaction_value), 0) AS net_revenue,
       COUNT(DISTINCT CASE WHEN is_return = 0 THEN order_id END) AS orders
FROM transactions
GROUP BY month ORDER BY month;

-- name: top_10_customers_by_spend
SELECT customer_id, ROUND(SUM(transaction_value), 0) AS net_spend,
       COUNT(DISTINCT CASE WHEN is_return = 0 THEN order_id END) AS orders
FROM transactions
GROUP BY customer_id ORDER BY net_spend DESC LIMIT 10;

-- name: repeat_purchase_rate
WITH per_customer AS (
  SELECT customer_id, COUNT(DISTINCT order_id) AS n_orders
  FROM transactions WHERE is_return = 0 GROUP BY customer_id)
SELECT COUNT(*) AS customers,
       SUM(n_orders >= 2) AS repeat_customers,
       ROUND(100.0 * SUM(n_orders >= 2) / COUNT(*), 1) AS repeat_rate_pct
FROM per_customer;

-- name: purchase_frequency_distribution
WITH per_customer AS (
  SELECT customer_id, COUNT(DISTINCT order_id) AS n_orders
  FROM transactions WHERE is_return = 0 GROUP BY customer_id)
SELECT CASE WHEN n_orders >= 10 THEN '10+' ELSE CAST(n_orders AS TEXT) END AS orders_bucket,
       COUNT(*) AS customers
FROM per_customer GROUP BY orders_bucket ORDER BY MIN(n_orders);

-- name: lapsed_share_by_rfm_segment
SELECT rfm_segment, COUNT(*) AS customers,
       ROUND(100.0 * AVG(status = 'Lapsed'), 1) AS lapsed_pct,
       ROUND(SUM(monetary), 0) AS revenue
FROM customers_scored GROUP BY rfm_segment ORDER BY revenue DESC;

-- name: high_value_high_risk
SELECT customer_id, rfm_segment, ROUND(churn_prob, 2) AS churn_prob, ROUND(annual_margin_run_rate, 0) AS annual_margin, priority
FROM customers_scored
WHERE status = 'Active' AND risk_level = 'High' AND value_tier = 'High'
ORDER BY margin_at_risk DESC LIMIT 10;
