from neo4j import GraphDatabase

uri='bolt://localhost:7687'
print('Trying no-auth connection to', uri)
try:
    driver = GraphDatabase.driver(uri)
    with driver.session() as s:
        rec = s.run('RETURN 1 AS ok').single()
        print('connected=', rec['ok'])
    driver.close()
except Exception as e:
    print('NO_AUTH_FAILED:', e)
