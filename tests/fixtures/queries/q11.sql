SELECT l.id AS lineitem_id, o.id AS order_id, p.name AS part_name, p.retail_price, l.quantity, l.extended_price
FROM lineitem l
INNER JOIN orders o ON l.order_id = o.id
INNER JOIN part p ON l.part_id = p.id
WHERE o.order_status = 'F' AND p.size > 10
ORDER BY l.quantity DESC, l.id ASC
LIMIT 15
