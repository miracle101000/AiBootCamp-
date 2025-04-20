import spacy 
from spacy import displaCy
import pandas as pd

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Sample text for NER
text = """
Amazon announced the launch of its new AI-powered logistics platform on April 12, 2025, during a press conference in Seattle. 
CEO Andy Jassy emphasized the importance of innovation in supply chain management. 
Meanwhile, competitors like Microsoft and Google are investing heavily in similar technologies. 
The project is expected to create over 5,000 jobs in the United States, particularly in cities like Austin, Texas and San Jose, California.
"""

# Process the text with spaCy
doc =  nlp(text)

# Function to extract entities
def extract_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'Entity': ent.text,
            'Label': ent.label_,
            'Explanation':spacy.explain(ent.label_)
        })
    return pd.DataFrame(entities)    

# Extract entities into a DataFrame
entities_df = extract_entities(doc)

# Display extracted entities to the user
print("Extracted Named Entities")
print(entities_df)

# Visualize Named Entities using DisplaCy
displaCy.render(doc, style="ent", jupyter=True)

# Save entities to a CSV file
entities_df.to_csv("extracted_entities.csv", index=False)
print("\nEntities saved 'extracted_entities.csv'")