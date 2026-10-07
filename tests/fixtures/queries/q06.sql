SELECT id, name, nation, phone, acctbal
FROM customer
WHERE acctbal < 0.0 AND NOT (phone IS NULL)
ORDER BY acctbal ASC
LIMIT 5
