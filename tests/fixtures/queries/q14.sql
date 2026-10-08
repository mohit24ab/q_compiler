SELECT nation, COUNT(*) AS cust_count, AVG(acctbal) AS avg_acctbal, MIN(acctbal) AS min_acctbal, MAX(acctbal) AS max_acctbal
FROM customer
GROUP BY nation
HAVING COUNT(*) > 3
