SELECT id, name, mfgr, brand, retail_price, retail_price * 1.15 AS taxed_price
FROM part
WHERE size <= 25 AND mfgr = 'Manufacturer#1'
ORDER BY retail_price ASC, id ASC
LIMIT 10
