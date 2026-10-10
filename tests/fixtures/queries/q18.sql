SELECT c.nation, COUNT(*) AS num_orders, SUM(o.total_price) AS total_spend, AVG(o.total_price) AS avg_order_spend
FROM customer c
INNER JOIN orders o ON c.id = o.cust_id
WHERE o.order_date >= '1995-01-01'
GROUP BY c.nation
HAVING SUM(o.total_price) > 10000.0
ORDER BY total_spend DESC, c.nation ASC
LIMIT 5
