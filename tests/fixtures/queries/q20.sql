SELECT c.mktsegment, c.nation, COUNT(*) AS item_count, SUM(l.extended_price) AS revenue
FROM customer c
INNER JOIN orders o ON c.id = o.cust_id
INNER JOIN lineitem l ON o.id = l.order_id
WHERE l.ship_date >= '1995-01-01' AND o.order_priority = '1-URGENT'
GROUP BY c.mktsegment, c.nation
HAVING SUM(l.extended_price) > 5000.0
ORDER BY revenue DESC, item_count DESC, c.mktsegment ASC, c.nation ASC
LIMIT 10
