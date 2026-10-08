SELECT id, order_id, part_id, quantity, extended_price, discount, tax
FROM lineitem
WHERE (discount >= 0.05 AND tax < 0.04) OR (quantity > 40 AND NOT (status = 'F'))
ORDER BY id ASC
LIMIT 25
