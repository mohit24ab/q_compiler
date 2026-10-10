SELECT id, cust_id, order_status, total_price, order_date
FROM orders
WHERE order_date >= '1995-01-01' AND order_status = 'O'
ORDER BY order_date ASC, total_price DESC, id ASC
LIMIT 15
