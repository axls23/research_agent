import os
from neo4j import GraphDatabase

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
pwd = os.getenv('NEO4J_PASSWORD')
db = os.getenv('NEO4J_DATABASE')

driver = GraphDatabase.driver(uri, auth=(user, pwd))
with driver.session(database=db) as s:
    print('--- labels ---')
    for r in s.run('CALL db.labels() YIELD label RETURN label ORDER BY label LIMIT 200'):
        print(r['label'])

    print('\n--- relationship types ---')
    for r in s.run('CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType LIMIT 200'):
        print(r['relationshipType'])

    print('\n--- indexes ---')
    for r in s.run('SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state RETURN name, type, entityType, labelsOrTypes, properties, state ORDER BY name LIMIT 300'):
        print(r['name'], '|', r['type'], '|', r['entityType'], '|', r['labelsOrTypes'], '|', r['properties'], '|', r['state'])

    print('\n--- node sample with most common label ---')
    top = s.run('MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC LIMIT 1').single()
    if top:
        lbl = top['l']
        print('top_label=', lbl, 'count=', top['c'])
        q = f"MATCH (n:`{lbl}`) RETURN properties(n) AS p LIMIT 1"
        rec = s.run(q).single()
        if rec:
            print('sample_props_keys=', sorted(list(rec['p'].keys()))[:80])
    else:
        print('No nodes in DB')

driver.close()
