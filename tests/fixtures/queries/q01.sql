SELECT id, name, nation, acctbal
FROM customer
WHERE mktsegment = 'BUILDING'
ORDER BY acctbal DESC, id ASC
LIMIT 10
