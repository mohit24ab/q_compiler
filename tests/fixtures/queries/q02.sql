SELECT id, order_id, quantity, extended_price, extended_price * (1.0 - discount) AS discounted_price
FROM lineitem
WHERE quantity > 10 AND return_flag = 'R'
ORDER BY extended_price DESC, id ASC
LIMIT 20
