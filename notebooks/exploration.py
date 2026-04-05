import pandas as pd

# Chargement des données
df = pd.read_csv('data/patients_dakar.csv')

print("--- APERÇU DES DONNÉES ---")
print(df)

# EXERCICE 1 (SECTION 10) : Analyse par sexe et par diagnostic
print("\n--- RÉSULTAT DE L'EXERCICE 1 : GROUPBY ---")
# Voici la ligne magique demandée :
analyse = df.groupby(['sexe', 'diagnostic']).size()

print(analyse)