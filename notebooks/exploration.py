import pandas as pd

# On charge les données qu'on a créées tout à l'heure
df = pd.read_csv("data/patients_dakar.csv")

print("-" * 40)
print("SENSANTE - EXPLORATION DES DONNÉES")
print("-" * 40)

print(f"\nNombre total de patients : {len(df)}")
print("\n--- Aperçu des 5 premiers patients ---")
print(df.head())

print("\n--- Statistiques de température ---")
print(f"Moyenne : {df['temperature'].mean():.1f}°C")
print(f"Maximum : {df['temperature'].max():.1f}°C")