SELECT l.id AS line_id, p.name AS part_name, p.brand, l.quantity, l.extended_price, p.retail_price
FROM lineitem l
INNER JOIN part p ON l.part_id = p.id
WHERE p.brand = 'Brand#11' AND l.quantity >= 20
ORDER BY l.extended_price DESC
LIMIT 15
