from gliner import GLiNER

# Charger le modèle pré-entraîné
model = GLiNER.from_pretrained("urchade/gliner_medium")

text = """
Monsieur Dupont, né le 10 janvier 1950, a reçu du paracétamol 1g
lors de la consultation du 12 mars 2023.
"""

labels = [
    "date_de_consultation",
    "age",
    "medicament"
]

entities = model.predict_entities(text, labels)

print("Entités détectées :")
for e in entities:
    print(e)
