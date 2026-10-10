SELECT p.brand, COUNT(*) AS line_count, SUM(l.quantity) AS total_qty, SUM(l.extended_price) AS total_revenue, AVG(l.discount) AS avg_discount
FROM lineitem l
INNER JOIN orders o ON l.order_id = o.id
INNER JOIN part p ON l.part_id = p.id
WHERE o.order_status = 'O' AND p.size > 5
GROUP BY p.brand
HAVING SUM(l.quantity) > 50
ORDER BY total_revenue DESC, p.brand ASC
LIMIT 10
