SELECT mfgr, brand, COUNT(*) AS part_count, MIN(retail_price) AS min_price, MAX(retail_price) AS max_price, AVG(retail_price) AS avg_price
FROM part
GROUP BY mfgr, brand
HAVING AVG(retail_price) > 1000.0 AND COUNT(*) >= 2
