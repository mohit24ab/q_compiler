SELECT order_status, order_priority, COUNT(*) AS order_count, SUM(total_price) AS total_revenue, AVG(total_price) AS avg_revenue
FROM orders
GROUP BY order_status, order_priority
HAVING SUM(total_price) > 50000.0
