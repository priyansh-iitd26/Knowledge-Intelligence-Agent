from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    )
)

with driver.session() as session:
    result = session.run("RETURN 'Neo4j Connected!' AS msg")
    print(result.single()["msg"])

driver.close()