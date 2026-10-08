SELECT l.id AS lineitem_id, o.id AS order_id, c.name AS cust_name, c.nation, l.extended_price
FROM lineitem l
INNER JOIN orders o ON l.order_id = o.id
INNER JOIN customer c ON o.cust_id = c.id
WHERE c.mktsegment = 'BUILDING' AND o.order_priority = '1-URGENT'
ORDER BY l.extended_price DESC
LIMIT 20
