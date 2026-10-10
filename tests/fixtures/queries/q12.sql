SELECT c.name AS cust_name, o.id AS order_id, l.id AS lineitem_id, l.extended_price, l.discount
FROM customer c
INNER JOIN orders o ON c.id = o.cust_id
INNER JOIN lineitem l ON o.id = l.order_id
WHERE c.nation = 'GERMANY' AND l.discount > 0.05
ORDER BY l.extended_price DESC, l.id ASC
LIMIT 10
