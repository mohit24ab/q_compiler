SELECT o.id AS order_id, c.name AS cust_name, c.nation, o.total_price, o.order_date
FROM orders o
INNER JOIN customer c ON o.cust_id = c.id
WHERE c.nation = 'UNITED STATES' AND o.total_price > 1000.0
ORDER BY o.total_price DESC
LIMIT 10
