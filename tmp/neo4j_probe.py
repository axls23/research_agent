import os
from neo4j import GraphDatabase

uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
user = os.getenv('NEO4J_USER', 'neo4j')
pw = os.getenv('NEO4J_PASSWORD', '')
print('NEO4J_URI=', uri)
print('NEO4J_USER=', user)
print('NEO4J_PASSWORD_SET=', bool(pw))

if not pw:
    raise SystemExit('Missing NEO4J_PASSWORD in environment')

driver = GraphDatabase.driver(uri, auth=(user, pw))
with driver.session() as s:
    rec = s.run('MATCH (n:PRISMAEntity) RETURN count(n) AS c').single()
    print('PRISMAEntity_count=', rec['c'])
    rec2 = s.run('MATCH (n:PRISMAEntity) WHERE n.embedding IS NOT NULL RETURN size(n.embedding) AS d LIMIT 1').single()
    print('embedding_dim_sample=', None if rec2 is None else rec2['d'])

driver.close()
