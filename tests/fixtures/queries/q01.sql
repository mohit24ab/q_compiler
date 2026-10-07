SELECT id, name, nation, acctbal
FROM customer
WHERE mktsegment = 'BUILDING'
ORDER BY acctbal DESC
LIMIT 10
