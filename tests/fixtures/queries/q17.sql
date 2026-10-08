SELECT ship_date, SUM(extended_price) AS daily_revenue, COUNT(*) AS daily_shipments
FROM lineitem
WHERE return_flag = 'N'
GROUP BY ship_date
HAVING COUNT(*) >= 2
