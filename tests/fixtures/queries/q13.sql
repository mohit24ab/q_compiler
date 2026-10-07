SELECT return_flag, status, SUM(quantity) AS sum_qty, SUM(extended_price) AS sum_base_price, AVG(discount) AS avg_disc, MIN(extended_price) AS min_price, MAX(extended_price) AS max_price, COUNT(*) AS count_order
FROM lineitem
GROUP BY return_flag, status
