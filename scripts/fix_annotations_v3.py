import json
import re

INPUT = "train_gliner_ready.json"
OUTPUT = "train_gliner_ready_v3.json"

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

def find(pattern, text, label):
    return [[m.start(), m.end(), label] for m in re.finditer(pattern, text, re.IGNORECASE)]

new_data = []

for sample in data:
    text = sample["text"]
    ner = []

    # Infos patient
    ner += find(r"\b\d{1,3}\s?ans\b", text, "age")
    ner += find(r"\b(homme|femme)\b", text, "sexe_du_patient")
    ner += find(r"\b(A|B|AB|O)[+-]\b", text, "groupe_sanguin_du_patient")
    ner += find(r"\b\d{1,2}\s\w+\s\d{4}\b", text, "date_de_consultation")

    # Symptômes / diagnostics
    ner += find(r"\b(fi[eè]vre|douleur|toux|fatigue|naus[eé]e)\b", text, "symptômes_signes_cliniques")
    ner += find(r"\b(di[aâ]b[eè]te|hypertension|asthme|grippe)\b", text, "pathologies_diagnostics")
    ner += find(r"\b(actuellement|en cours)\b.*\b(di[aâ]b[eè]te|hypertension)\b", text, "pathologies_diagnostics_actuels")
    ner += find(r"\b(ant[eé]c[eé]dent[s]?)\b.*\b(di[aâ]b[eè]te|hypertension)\b", text, "pathologies_diagnostics_antécédents")

    # Médicaments
    ner += find(r"\b(parac[eé]tamol|ibuprof[eè]ne|amoxicilline)\b", text, "médicaments_molécules")
    ner += find(r"\b(parac[eé]tamol|ibuprof[eè]ne)\b.*\bprescrit\b", text, "médicaments_molécules_prescrits")
    ner += find(r"\b(parac[eé]tamol|ibuprof[eè]ne)\b.*\badministr[eé]\b", text, "médicaments_molécules_administrés")

    # Posologie / administration
    ner += find(r"\b\d+\s?(mg|g|ml)\b", text, "posologie_dosage_quantité")
    ner += find(r"\b(une fois|deux fois|trois fois)\spar\sjour\b", text, "posologie_fréquence")
    ner += find(r"\b(voie orale|intraveineuse|IM)\b", text, "administration")

    # Contexte médical
    ner += find(r"\b(allergie|hypersensibilit[eé])\b.*", text, "allergies_et_hypersensibilités")
    ner += find(r"\b(ant[eé]c[eé]dent[s]?)\b", text, "antécédents")
    ner += find(r"\b(vie|travail|domicile|famille)\b", text, "contexte_de_vie")
    ner += find(r"\btraitement\b.*", text, "traitements")
    ner += find(r"\b(pacemaker|proth[eè]se|sonde)\b", text, "dispositifs_médicaux")

    if ner:
        new_data.append({"text": text, "ner": ner})

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"✅ Dataset corrigé généré : {OUTPUT}")
print(f"📄 Nombre de samples : {len(new_data)}")
