SELECT l.id AS lineitem_id, o.id AS order_id, o.order_date, l.ship_date, l.extended_price
FROM lineitem l
INNER JOIN orders o ON l.order_id = o.id
WHERE o.order_status = 'O' AND l.ship_date > o.order_date
ORDER BY l.ship_date ASC
LIMIT 20
