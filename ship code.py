from xml.parsers.expat import model
import pandas as pd
import numpy as np
from prompt_toolkit import Application
import xgboost as xgb
import matplotlib.pyplot as plt
import  sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# 1. Chargement du fichier avec le bon séparateur
df = pd.read_csv('data_navire.csv', sep=';')
# Nettoyage des noms de colonnes (enlever les espaces)
df.columns = df.columns.str.strip()
# Prévisualisation
df.head(10)
df.info()
# Nettoyage des données
# Suppression des lignes avec des valeurs manquantes dans les colonnes critiques
df['INSURED VALUE'] = df['INSURED VALUE'].astype(str).str.replace(r'[^\d.]', '', regex=True)
df['INSURED VALUE'] = pd.to_numeric(df['INSURED VALUE'], errors='coerce')
df.head(10)
df.info()
# 2. Nettoyage de la cible (Suppression des virgules et conversion)
if df['INSURED VALUE'].dtype == 'O':
    df['INSURED VALUE'] = df['INSURED VALUE'].str.replace(',', '').astype(float)

# creation d'une nouvelle colonne age des navires
df['AGE'] = 2026 - df['BUILT']
# Nettoyage des variables catégorielles
df['TYPE'] = df['TYPE'].str.upper().str.strip()
# Prévisualisation après nettoyage
df.head(10)
df.info()
# Création d'un dictionnaire de remplacement pour fusionner les synonymes
# Fusion des catégories similaires dans la colonne 'TYPE'
mappage = {
    'TANKER': 'OIL TANKER',
    'OIL PRODUCTS TANKER': 'OIL TANKER',
    'OIL PRODUCTS TAN': 'OIL TANKER',
    'OFFSHORE SUPPI': 'OFFSHORE SUPPLY',
    'OFFSHORE SUPPL': 'OFFSHORE SUPPLY',
    'MPP BULK': 'BULK CARRIER',
    'BBU': 'BULK CARRIER',
    'GENERAL CARGO SH': 'GENERAL CARGO',
    'GENERAL CARGO VESSEL': 'GENERAL CARGO',
    'CONTAINER SHIP': 'CONTAINER',
    'CONTAINER VESSEL': 'CONTAINER',
    'UCC': 'CONTAINER',
    'dredge': 'DREDGER',
    'lpg tanker': 'LPG TANKER',
    'lpg': 'LPG TANKER',
}
# Application du remplacement
df['TYPE'] = df['TYPE'].replace(mappage)
# Vérification : Afficher le nombre de navires par catégorie nettoyée
print("Nouvelles catégories unifiées :")
print(df['TYPE'].value_counts())
# ENCODAGE DU TEXTE (TYPE)
# On transforme les catégories des navires en numéros
le = LabelEncoder()
df['TYPE_ENCODED'] = le.fit_transform(df['TYPE'])
# Vérification de l'encodage
print("Encodage des types de navires :")
print(df['TYPE_ENCODED'].head(10))
# type navire inconnus pour le modèle
print("Types de navires uniques après encodage :")
print(df['TYPE'].unique())
# Renommage des colonnes en français
# Dictionnaire de traduction des colonnes
traductions = {
    'IMO': 'IMO',
    'TYPE': 'Type_Navire',
    'BUILT': 'Annee_Construction',
    'GRT': 'GRT',
    'DWT': 'DWT',
    'Engine power': 'Puissance_Moteur',
    'Builder': 'Constructeur',
    'CLASS': 'Societe_Classification',
    'FLAG': 'Pavillon',
    'INSURED VALUE': 'Valeur_Assuree'
}
# Application du renommage au DataFrame
df = df.rename(columns=traductions)
# Affichage pour vérifier
print("Colonnes renommées :")
print(df.columns.tolist())
# Supprime la colonne 'CAT CLASS' on replace par regroupement par 'Class' avec création d'une nouvelle colonne
df = df.drop(columns=['CAT CLASS'])

# Liste des membres principaux de l'IACS (International Association of Classification Societies)
iacs_members = ['ABS', 'BV', 'DNV', 'LR', 'NK', 'RINA', 'KR', 'CCS', 'RS', 'CRS', 'PRS']
# 4. Affichage de la liste des classes de sociétés de classification
print("Liste des Classes:")
print(df['Societe_Classification'].value_counts())
#Nettoyage colonne Societe_Classification
df['Societe_Classification'] = df['Societe_Classification'].astype(str).str.strip().str.upper()
# Fusion des classes similaires
mappage_unification = {
    'NKK': 'NK',
    'CSC': 'CCS',
    'RMRS': 'RS',
    'TURK LLOYD': 'TURKISH LLOYD',
}
df['Societe_Classification'] = df['Societe_Classification'].replace(mappage_unification)
# Vérification après nettoyage
print("Liste des Classes:")
print(df['Societe_Classification'].value_counts())

# Création d'une nouvelle colonne binaire
df['is_IACS'] = df['Societe_Classification'].isin(iacs_members).astype(int)
# Vérification
print("Vérification de la colonne is_IACS :")
print(df.columns.tolist())
print(df['is_IACS'].value_counts())


#Analyse statistique détaillée par Classe membres IACS
# On analyse uniquement les sociétés membres de l'IACS
df_iacs = df[df['Societe_Classification'].isin(iacs_members)]
# Calcul des statistiques par société
analyse_classe = df_iacs.groupby('Societe_Classification')['Valeur_Assuree'].agg(['mean', 'median', 'count', 'std']).sort_values(by='mean', ascending=False)
# Affichage des résultats
print("Analyse détaillée des membres IACS :")
print(analyse_classe)
# Visualisation des différences de valeur assurée entre les sociétés membres de l'IACS
# Boxplot
plt.figure(figsize=(15,7))
sns.boxplot(x='Societe_Classification', y='Valeur_Assuree', data=df_iacs)
plt.xticks(rotation=45)
plt.title('Comparaison de la Valeur Assurée entre les membres de l\'IACS')
plt.show()
# Visualisation des différences de valeur assurée entre types de navires
# Boxplot
plt.figure(figsize=(15,7))
sns.boxplot(x='TYPE_ENCODED', y='Valeur_Assuree', data=df_iacs)
plt.xticks(rotation=45)
plt.title('Comparaison de la Valeur Assurée entre les Types de Navires (Membres IACS)')
plt.show()
# Visualisation du ratio Valeur/DWT par Société de Classification
plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x='DWT', y='Valeur_Assuree', hue='Societe_Classification', style='Societe_Classification', s=100)
plt.title('Relation Valeur vs DWT par Classe')
plt.grid(True)
plt.show()
# Création du ratio valeur par Tonne (DWT)
df['Ratio_Valeur_DWT'] = df['Valeur_Assuree'] / df['DWT']
# Prévisualisation
df[['Valeur_Assuree', 'DWT', 'Ratio_Valeur_DWT']].head(10)
# Visualisation du ratio Valeur/DWT par Type de Navire avec Boxplot et Strip
plt.figure(figsize=(16, 8))
sns.boxplot(x='Type_Navire', y='Ratio_Valeur_DWT', data=df, palette='Set3', showfliers=True)
# Le boxplot montre les quartiles et les moustaches
sns.stripplot(x='Type_Navire', y='Ratio_Valeur_DWT', data=df, color='black', size=4, alpha=0.3)
# Le stripplot superpose les points individuels pour visualiser la distribution
plt.title('Identification des Outliers : Ratio Valeur/DWT par Type de Navire', fontsize=15)
plt.xlabel('Type de Navire', fontsize=12)
plt.ylabel('Ratio Valeur / DWT (USD par Tonne)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Fonction pour filtrer les outliers par groupe (Type_Navire)
def filter_outliers_by_type(group):
    # On définit les seuils bas et haut (1.5 * IQR est le standard)
    Q1 = group['Ratio_Valeur_DWT'].quantile(0.25)
    Q3 = group['Ratio_Valeur_DWT'].quantile(0.75)
    IQR = Q3 - Q1
    # On définit les seuils bas et haut (1.5 * IQR est le standard)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # On retourne le groupe filtré
    return group[(group['Ratio_Valeur_DWT'] >= lower_bound) & (group['Ratio_Valeur_DWT'] <= upper_bound)]
#Application de la fonction à chaque groupe de Type_Navire
df_final = df.groupby('Type_Navire', group_keys=False).apply(filter_outliers_by_type)
# Vérification après suppression des outliers
print("Taille du DataFrame après suppression des outliers :", df_final.shape)
# Visualisation du ratio Valeur/DWT par Type de Navire après suppression des outliers
plt.figure(figsize=(16, 8))
sns.boxplot(x='Type_Navire', y='Ratio_Valeur_DWT', data=df_final, palette='viridis')
plt.xticks(rotation=45)
plt.title('Vérification : Ratio Valeur/DWT par Type après suppression des Outliers')
plt.grid(axis='y', alpha=0.3)
plt.show()

# Application du plafonnement manuel après le filtrage IQR
df_final = df_final[df_final['Ratio_Valeur_DWT'] < 10000]
print("Taille du DataFrame après plafonnement manuel :", df_final.shape)
print ('nombre de navires final :', len (df_final))
# Visualisation des corrélations après nettoyage
plt.figure(figsize=(10,8))
sns.boxplot(x='Type_Navire', y='Ratio_Valeur_DWT', data=df_final, palette='viridis')
plt.xticks(rotation=45)
plt.title('Vérification : Ratio Valeur/DWT par Type après suppression des Outliers')
plt.grid(axis='y', alpha=0.3)
plt.show()
# Influence sur la Valeur Assurée
# Calcul de la matrice de corrélation 
# Sélection des colonnes numériques pertinentes
cols_finales = ['Valeur_Assuree', 'AGE', 'DWT', 'GRT', 'Puissance_Moteur', 'Ratio_Valeur_DWT']

# Calcul de la matrice de corrélation et affichage
plt.figure(figsize=(12, 10))
correlation_matrix = df_final[cols_finales].corr()
# Affichage de la Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
plt.title('Quelle variable influence le plus la Valeur Assurée (Données Nettoyées)? ', fontsize=15)
plt.show()

# Influence de la Zone de Construction sur la Valeur Assurée
# Nettoyage et mise en majuscules
df_final['Constructeur'] = df_final['Constructeur'].str.strip().str.upper()
print("Pays de Construction :")
print(df_final['Constructeur'].unique())
#corection des pays de construction
df_final['Pays_Construction'] = df_final['Constructeur'].replace({'ÉTATS-UNIS': 'USA'})

# Encodage numérique des pays de construction
le_pays = LabelEncoder()
df_final['PAYS_ENC'] = le_pays.fit_transform(df_final['Pays_Construction'])

# Affichage pour vérification de l'encodage
print("Mapping des pays encodés :")
for index, class_label in enumerate(le_pays.classes_):
    print(f"{index} : {class_label}")
#visualisation l'influence sur la Valeur Assurée ajout de la pays de construction
cols_finales = ['Valeur_Assuree', 'AGE', 'DWT', 'GRT', 'Puissance_Moteur', 'PAYS_ENC', 'Ratio_Valeur_DWT']

# Calcul de la matrice de corrélation et affichage
plt.figure(figsize=(12, 10))
correlation_matrix = df_final[cols_finales].corr()
# Affichage de la Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
plt.title('Quelle variable influence le plus la Valeur Assurée ajout de la variable Pays de Construction?', fontsize=15)
plt.show()
# ==========================================
# 3. PRÉPARATION DES DONNÉES ET MODÈLE
# ==========================================
# 1. Définition des entrées (X) et de la cible (y)
# On retire les colonnes non-prédictives
features = ['AGE', 'DWT', 'GRT', 'Puissance_Moteur', 'TYPE_ENCODED', 'is_IACS', 'PAYS_ENC']
X = df_final[features]
y = np.log1p(df_final['Valeur_Assuree']) # Transformation log pour stabiliser la variance
# 2. Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialisation et entraînement du modèle XGBoost
model_final = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    objective='reg:squarederror'
    
)
model_final.fit(X_train, y_train)
# ==========================================
# 4. ÉVALUATION ET INTERPRÉTABILITÉ (XAI)
# ==========================================
# 1. Prédiction sur l'échelle Log
y_pred_log = model_final.predict(X_test)

# 2. Retour à l'échelle réelle ( Dollars)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)
print(f"✅ Modèle entraîné avec succès !")
# 3. Calcul des métriques
r2 = r2_score(y_test, y_pred_log) # Le R² peut se calculer sur le log
mae = mean_absolute_error(y_test_real, y_pred_real) # La MAE doit être sur le réel
# 4. Affichage des résultats
print(f"Précision (R²) : {r2:.2%}")
print(f"Erreur moyenne (MAE) : {mae:,.2f} dollars")
# 1. Création du tableau de comparaison
comparison_df = pd.DataFrame({
    'Valeur Réelle (USD)': y_test_real.values,
    'Valeur Prédite (USD)': y_pred_real,
    'Erreur Absolue (USD)': np.abs(y_test_real - y_pred_real)
})

# 2. Ajout de l'erreur en pourcentage pour mieux juger la précision
comparison_df['Erreur %'] = (comparison_df['Erreur Absolue (USD)'] / comparison_df['Valeur Réelle (USD)']) * 100

# 3. Affichage des 10 premières lignes
print("--- Comparaison des résultats sur le jeu de test ---")
print(comparison_df.head(10).to_string(formatters={'Valeur Réelle (USD)': '{:,.0f}'.format, 
                                                   'Valeur Prédite (USD)': '{:,.0f}'.format, 
                                                   'Erreur Absolue (USD)': '{:,.0f}'.format,
                                                   'Erreur %': '{:.2f}%'.format}))

# 4. Calcul de la fiabilité globale
fiabilite = 100 - comparison_df['Erreur %'].mean()
print(f"\nFiabilité moyenne du modèle : {fiabilite:.2f}%")

# Analyse spécifique du navire n°73 (le plus gros échec)
error_73 = comparison_df.loc[73]
details_73 = df_final.loc[73]

print("--- ANALYSE DU NAVIRE N°73 ---")
print(f"Caractéristiques : Age={details_73['AGE']}, DWT={details_73['DWT']}, Type={details_73['Type_Navire']}")
print(f"Erreur : {error_73['Erreur %']:.2f}%")

# Visualisation de l'importance des variables
plt.figure(figsize=(10,5))
xgb.plot_importance(model_final, importance_type='weight')
plt.title("Quelles variables l'IA a-t-elle trop écouté ?")
plt.show()

# ==========================================
# 3. CODE DE VÉRIFICATION (Nouveau)
# ==========================================
print("\n--- ANALYSE DE VÉRIFICATION ---")

# A. Importance des Variables (Pour voir si le modèle est logique)

plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model_final.feature_importances_, index=features)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')
plt.title('Importance des variables : Qu\'est-ce qui influence le prix ?')
plt.show()

# B. Analyse des Résidus (Vérifier si le modèle fait des erreurs systématiques)

preds_log = model_final.predict(X_test)
residuals = y_test - preds_log
plt.figure(figsize=(10, 5))
sns.scatterplot(x=np.expm1(preds_log), y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Analyse des Résidus : Le modèle est-il biaisé ?')
plt.xlabel('Valeurs Prédites')
plt.ylabel('Erreur (Log)')
plt.show()

# C. Validation Croisée (Vérifier la stabilité sur différentes coupes de données)
cv_scores = cross_val_score(model_final, X, y, cv=5)
print(f"Stabilité du modèle (CV Score moyen) : {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.2f})")

# D. Interprétabilité (PDP)
# Correction : utiliser les indices des colonnes pour éviter les erreurs de noms
fig, ax = plt.subplots(figsize=(15, 5))
PartialDependenceDisplay.from_estimator(
    model_final, 
    X_train, 
    features=[0, 1, 5], # AGE, DWT, is_IACS
    feature_names=features,
    ax=ax
)
plt.suptitle('Analyse PDP : Impact isolé de l\'Age et du Tonnage')
plt.show()

# ==========================================
# 4. TEST DE PRÉDICTION MANUELLE (Vérification finale)
# ==========================================
print("\n--- TEST SUR UN NAVIRE FICTIF ---")
# On crée un navire type : 10 ans, 5000 DWT, 2500 GRT, 1200 Power, Type 5, IACS=1, Pays 1
fake_ship = np.array([[10, 5000, 2500, 1200, 5, 1, 1]])
prediction = np.expm1(model_final.predict(pd.DataFrame(fake_ship, columns=features)))
print(f"Prix estimé pour ce navire : {prediction[0]:,.0f} USD")

# --- Bonus : Importance des Variables ---
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', color='teal')
plt.title("Quelles caractéristiques influencent le plus le prix ?")
plt.xlabel("Poids relatif dans l'algorithme")
plt.show()

2222

#Configuration du Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

# 2. Entraînement
rf_model.fit(X_train, y_train)

# 3. Prédictions et conversion (Log -> Réel)
y_pred_rf_log = rf_model.predict(X_test)
y_pred_rf_real = np.expm1(y_pred_rf_log)

# 4. Calcul des métriques
r2_rf = r2_score(y_test, y_pred_rf_log)
mae_rf = mean_absolute_error(y_test_real, y_pred_rf_real)

print(f"--- RÉSULTATS RANDOM FOREST ---")
print(f"Précision (R²) : {r2_rf:.2%}")
print(f"Erreur moyenne (MAE) : {mae_rf:,.2f} USD")
#---------------------------------------------------
    # Comparaison des performances entre XGBoost et Random Forest
#-----------------------------------------------
models = ['XGBoost', 'Random Forest']
r2_scores = [75.32, 68.23]
mae_values = [1718669, 1804749]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Axe pour le R2
color = 'tab:blue'
ax1.set_xlabel('Modèles')
ax1.set_ylabel('R² Score (%)', color=color)
ax1.bar(models, r2_scores, color=color, alpha=0.6, width=0.4, label='R² (%)')
ax1.tick_params(axis='y', labelcolor=color)

# Axe pour la MAE
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('MAE (USD)', color=color)
ax2.plot(models, mae_values, color=color, marker='o', linewidth=3, label='MAE')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Comparaison de Performance : XGBoost vs Random Forest')
plt.show()

