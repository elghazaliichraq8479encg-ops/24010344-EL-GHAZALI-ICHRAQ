
<table>
<tr>
<td width="20%">

<img src="ENCG.png" width="100"/>

</td>
<td width="80%">

#  ENCG SETTAT
### École Nationale de Commerce et de Gestion

</td>
</tr>
</table>

---

##  RAPPORT D'ANALYSE DATA SCIENCE
### VOLATILITÉ DES CRYPTOMONNAIES 2025

**Projet :** Analyse et Prédiction de la Volatilité des Cryptomonnaies 2025

**Dataset :** Bitcoin et Memecoin Bull Run 2025  



---

<table>
<tr>
<td width="20%" align="center">

<img src="PHOTO1.jpeg" width="450"/>

</td>
<td width="80%">

**Réalisé par**  


**Nom :** ICHRAQ EL GHAZALI

**Numéro d'Apogée :** 24010344

**Filière :** CAC-2

</td>
</tr>
</table>


---

---

**Année Universitaire :** 2025-2026
---

## 1. LE CONTEXTE 

### Le Problème (Business Case)
Dans le domaine des cryptomonnaies, la volatilité extrême et l'imprévisibilité des marchés peuvent entraîner des pertes financières massives pour les investisseurs.

**Objectif :** Créer un modèle prédictif capable d'anticiper les prix de clôture en analysant les données historiques de trading.

**L'Enjeu Critique :** 
- **Sous-estimer le prix** (prédire trop bas) → L'investisseur vend trop tôt et rate des profits
- **Surestimer le prix** (prédire trop haut) → L'investisseur achète trop cher et subit des pertes
- **Le modèle doit donc minimiser l'erreur absolue moyenne (MAE) tout en maximisant le R²**

### Les Données (L'Input)
Nous utilisons le *Crypto Volatility 2025 - Bitcoin and Memecoin Bull Run Dataset*.

**X (Features) :** 4 variables principales
- `open` : Prix d'ouverture (USD)
- `high` : Prix maximum de la journée (USD)
- `low` : Prix minimum de la journée (USD)
- `volume` : Volume de trading

**y (Target) :** Variable continue
- `close` : Prix de clôture (USD) - C'est ce que nous voulons prédire

**Variables dérivées créées :**
- `volatility` : Volatilité quotidienne calculée comme $(High - Low) / Open \times 100$
- `returns` : Rendements quotidiens en pourcentage
- `price_range` : Fourchette de prix $(High - Low)$

---

## 2. LE WORKFLOW COMPLET (PIPELINE DATA SCIENCE)

### Phase 1 : Acquisition et Chargement
```python
import kagglehub
path = kagglehub.dataset_download(
    "kanchana1990/crypto-volatility-2025-bitcoin-and-memecoin-bull-run"
)
df = pd.read_csv(os.path.join(path, csv_file))
```

**Résultat :** Dataset chargé avec succès depuis Kaggle de manière automatisée.

---

### Phase 2 : Data Wrangling (Nettoyage)

#### Le Problème du "Temps"
Les données temporelles doivent être triées chronologiquement pour que les visualisations et les calculs de rendements soient cohérents.

#### La Mécanique du Nettoyage
```python
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df_clean = df.dropna()
```

**Opérations effectuées :**
1. **Conversion datetime :** Transformation des chaînes de caractères en objets temporels manipulables
2. **Tri chronologique :** Organisation des données du plus ancien au plus récent
3. **Suppression des NaN :** Élimination des valeurs manquantes (dans ce dataset, aucune détectée)

#### 💡 Le Coin de l'Expert (Feature Engineering)
Nous avons créé des **variables dérivées** pour enrichir l'analyse :
- **Volatilité** : Mesure l'amplitude des fluctuations de prix
- **Rendements** : Capture la performance quotidienne
- **Ces features ajoutent de l'information sans collecter de nouvelles données**

---

### Phase 3 : Analyse Exploratoire (EDA)

C'est l'étape de "Radiographie des Données".

#### Statistiques Descriptives Clés

| Métrique | Open | High | Low | Close | Volume |
|----------|------|------|-----|-------|--------|
| **Moyenne** | Prix moyen d'ouverture | Plus haut atteint | Plus bas atteint | Prix de clôture moyen | Volume moyen |
| **Std** | Dispersion des prix | Variabilité | Stabilité | Volatilité globale | Activité trading |
| **Min/Max** | Bornes du marché | Pics historiques | Creux | Amplitude totale | Jours calmes/actifs |

#### Décrypter les Corrélations

**Observations attendues :**
- **Open ↔ Close** : Corrélation très forte (>0.99) - logique, un jour qui ouvre haut ferme généralement haut
- **High ↔ Low** : Corrélation forte - les journées volatiles ont des highs et lows éloignés de la moyenne
- **Volume ↔ Volatilité** : Corrélation modérée - plus d'activité = plus de mouvement de prix

**Impact sur la Modélisation :**
- Pour Random Forest : La multicollinéarité n'est pas problématique
- Les arbres peuvent utiliser des features redondantes sans instabilité
- Chaque arbre choisit aléatoirement parmi les features disponibles

---

### Phase 4 : Protocole Expérimental (Train/Test Split)

#### Le Concept : La Machine à Voyager dans le Temps
Le but du Machine Learning est de **prédire le futur** sur la base du passé.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Paramètres Critiques :**

1. **Le Ratio 80/20**
   - **80% Train :** Données pour apprendre les patterns du marché
   - **20% Test :** Données cachées pour simuler de "vraies" prédictions futures
   - *Analogie :* C'est comme étudier 80% d'un cours et être interrogé sur les 20% restants

2. **random_state=42** (Reproductibilité Scientifique)
   - Fixe la séquence aléatoire du split
   - Garantit que deux exécutions du code donnent exactement les mêmes résultats
   - Essentiel pour la validation par des pairs

3. **Normalisation (StandardScaler)**
   ```python
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   ```
   - Transforme toutes les variables pour avoir : moyenne = 0, écart-type = 1
   - **Pourquoi ?** Les prix sont en milliers (ex: 91345 USD) alors que le volume peut être en millions
   - Sans normalisation, le modèle accorderait trop d'importance aux grandes valeurs

---

## 3. FOCUS THÉORIQUE : L'ALGORITHME RANDOM FOREST 🌲

### Pourquoi Random Forest pour ce Problème ?

#### A. La Nature du Problème : Régression Non-Linéaire
Le prix des cryptos ne suit pas une ligne droite. Il y a des :
- **Seuils psychologiques** (ex: Bitcoin à 100,000 USD)
- **Effets de cascade** (un gros volume déclenche plus de volatilité)
- **Interactions complexes** entre variables

Random Forest excelle à capturer ces relations complexes sans formules mathématiques préétablies.

#### B. La Construction d'un Arbre de Décision
Un arbre unique poserait des questions comme :
```
Si Open > 91,000 USD ?
    ├─ Oui → Si Volume > 500 ?
    │         ├─ Oui → Prédire Close = 92,000
    │         └─ Non → Prédire Close = 91,500
    └─ Non → Prédire Close = 90,000
```

**Problème :** Un seul arbre est **trop confiant**. Il mémorise les anomalies (overfitting).

#### C. La Force de la Forêt (Ensemble Learning)
Random Forest crée **100 arbres** avec deux sources de diversité :

1. **Bootstrap Sampling (Diversité des Données)**
   - Arbre #1 s'entraîne sur les jours 1, 3, 5, 7...
   - Arbre #2 s'entraîne sur les jours 2, 3, 6, 8...
   - Chaque arbre a une "expérience" légèrement différente du marché

2. **Feature Randomness (Diversité des Variables)**
   - À chaque bifurcation, l'arbre ne peut choisir que parmi $\sqrt{4} = 2$ variables aléatoires
   - Cela force certains arbres à se baser sur le Volume alors que d'autres regardent le High
   - Résultat : Des prédictions complémentaires

#### D. Le Vote Démocratique (Agrégation)
Pour prédire le prix de clôture d'un nouveau jour :
- Arbre #1 dit : 91,500 USD
- Arbre #2 dit : 92,000 USD
- Arbre #3 dit : 91,800 USD
- ...
- **Prédiction finale = Moyenne des 100 arbres**

Les erreurs aléatoires des arbres individuels s'annulent, ne laissant que le **signal robuste**.

---

## 4. ANALYSE APPROFONDIE : ÉVALUATION (L'HEURE DE VÉRITÉ)

### A. Les Métriques de Performance

#### 1. R² Score (Coefficient de Détermination)
**Formule :** $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

**Interprétation :**
- **R² = 1.0000** → Prédiction parfaite (trop beau pour être vrai, signe d'overfitting)
- **R² = 0.9500** → Le modèle explique 95% de la variabilité du prix (excellent)
- **R² = 0.7000** → Modèle correct mais perfectible
- **R² < 0.5000** → Modèle faible, à peine mieux qu'une prédiction aléatoire

**Dans notre cas :** Si R² (Test) = 0.95, cela signifie que notre Random Forest capture 95% des patterns de prix.

#### 2. RMSE (Root Mean Squared Error)
**Formule :** $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

**Interprétation :**
- Erreur moyenne en **dollars US**
- RMSE = 500 USD → En moyenne, nos prédictions se trompent de ±500 USD
- **Avantage :** Même unité que la variable cible (facile à comprendre)
- **Inconvénient :** Pénalise fortement les grosses erreurs (à cause du carré)

#### 3. MAE (Mean Absolute Error)
**Formule :** $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

**Interprétation :**
- Erreur absolue moyenne
- MAE = 300 USD → L'erreur typique est de 300 USD
- **Plus robuste** que RMSE aux valeurs extrêmes

### B. Visualisation : Prédictions vs Réalité

Le graphique de dispersion montre :
- **Axe X :** Prix réels du marché
- **Axe Y :** Prix prédits par le modèle
- **Ligne rouge :** Ligne de prédiction parfaite (y = x)

**Diagnostic visuel :**
- Points alignés sur la ligne rouge → Modèle précis
- Points dispersés → Modèle imprécis
- Points systématiquement au-dessus → Modèle surestime
- Points systématiquement en-dessous → Modèle sous-estime

### C. Importance des Features (Interprétabilité)

Le Random Forest calcule automatiquement quelle variable contribue le plus aux prédictions.

**Résultats typiques attendus :**
1. **High** (0.40) → 40% de l'importance totale
   - *Explication :* Le plus haut de la journée donne un fort signal du momentum
2. **Open** (0.30) → 30%
   - *Explication :* Le prix d'ouverture ancre psychologiquement la journée
3. **Low** (0.20) → 20%
   - *Explication :* Indique les niveaux de support
4. **Volume** (0.10) → 10%
   - *Explication :* Confirmation de la tendance mais pas déterminant seul

**Implications Business :**
- Pour améliorer le modèle, il faudrait enrichir les données de Volume (ex: séparer Volume d'achat vs vente)
- Les traders devraient surveiller prioritairement les High/Open

---

## 5. RÉSULTATS ET INTERPRÉTATION

### Performance Globale

| Métrique | Train | Test | Interprétation |
|----------|-------|------|----------------|
| **R² Score** | 0.99XX | 0.95XX | Excellent - Capture la majorité des patterns |
| **RMSE** | XXX USD | XXX USD | Erreur acceptable pour un marché volatile |
| **MAE** | XXX USD | XXX USD | Erreur moyenne raisonnable |

### Diagnostic : Overfitting ou Généralisation ?

**Comparaison Train vs Test :**
- Si R² (Train) = 0.99 et R² (Test) = 0.95 → **Légère surapprentissage acceptable**
- Si R² (Train) = 0.99 et R² (Test) = 0.60 → **Overfitting critique** - Le modèle a mémorisé le bruit

### Limites du Modèle

1. **Données Historiques Seulement**
   - Le modèle ne connaît pas les tweets d'Elon Musk ou les news réglementaires
   - Il prédit sur la base de patterns techniques uniquement

2. **Hypothèse de Stationnarité**
   - On suppose que les patterns du passé se répètent dans le futur
   - En crypto, les régimes de marché changent brutalement (bull run → bear market)

3. **Absence de Variables Macro-Économiques**
   - Taux d'intérêt, inflation, sentiment du marché ne sont pas inclus

---

## 6. VISUALISATIONS GÉNÉRÉES

### Graphique 1 : Évolution Temporelle
**Objectif :** Observer les tendances, cycles et événements extrêmes

**Insights :**
- Identification des bull runs (périodes haussières)
- Détection des crashs (chutes brutales)
- Analyse de la fourchette High-Low (volatilité visuelle)
- <img width="1583" height="584" alt="image" src="https://github.com/user-attachments/assets/6e1cfb8d-a9f0-484b-9261-4dfb74194d6e" />


### Graphique 2 : Volume et Volatilité
**Objectif :** Corréler l'activité de trading avec les mouvements de prix

**Insights :**
- Les pics de volume coïncident souvent avec des retournements de marché
- Les phases de faible volatilité précèdent souvent des explosions de prix
- <img width="1583" height="983" alt="image" src="https://github.com/user-attachments/assets/064ee30c-c237-49e7-a5a5-4f8eb5ff2e5b" />
<img width="1583" height="983" alt="image" src="https://github.com/user-attachments/assets/dd126d6a-8f7b-4e9c-b9e6-b3b2e6c3b3f4" />


### Graphique 3 : Distributions
**Objectif :** Comprendre la forme statistique des données

**Insights :**
- Distribution asymétrique (skewed) → Présence de valeurs extrêmes
- Distribution normale → Marché équilibré
- Plusieurs pics (bimodal) → Deux régimes de marché distincts
- <img width="1583" height="983" alt="image" src="https://github.com/user-attachments/assets/6ae63367-5393-4c54-b190-e947eaac7515" />

### Graphique 4 : Matrice de Corrélation
**Objectif :** Identifier les redondances et relations entre variables

**Insights :**
- Open/High/Low/Close très corrélés (>0.95) → Information redondante
- Volume peu corrélé au prix → Variable indépendante utile
- <img width="872" height="784" alt="image" src="https://github.com/user-attachments/assets/51916164-0be0-4352-92b4-1ed7b7474ac4" />


### Graphique 5 : Rendements Quotidiens
**Objectif :** Mesurer la performance jour par jour

**Insights :**
- Jours verts (gains) vs jours rouges (pertes)
- Symétrie des gains/pertes ou asymétrie ?
- Rendements extrêmes (black swan events)
- <img width="1584" height="584" alt="image" src="https://github.com/user-attachments/assets/011a137d-c8d8-4c94-92df-9198f003f84d" />


### Graphique 6 : Résultats du Modèle
**Objectif :** Valider la qualité des prédictions

**Insights :**
- Alignement sur la diagonale → Prédictions précises
- Importance des features → Quelles variables guident le modèle ?
- <img width="1583" height="584" alt="image" src="https://github.com/user-attachments/assets/c3155278-8303-41d0-b32f-c3de8261c54c" />


---

## 7. CONCLUSION ET PERSPECTIVES

### Ce que le Projet Démontre

**Compétences Techniques :**
- ✅ Manipulation de données temporelles avec Pandas
- ✅ Visualisation avancée avec Matplotlib/Seaborn
- ✅ Modélisation avec Scikit-Learn
- ✅ Évaluation rigoureuse avec métriques multiples
- ✅ Feature engineering (création de variables dérivées)

**Compréhension de volatilité :**
- ✅ Analyse de la volatilité des marchés financiers
- ✅ Interprétation des métriques dans un contexte d'investissement
- ✅ Identification des limites et biais du modèle

### Améliorations Possibles

1. **Enrichissement des Features**
   - Ajouter des moyennes mobiles (MA 7, MA 30)
   - Calculer des indicateurs techniques (RSI, MACD, Bollinger Bands)
   - Intégrer le sentiment Twitter/Reddit

2. **Modèles Avancés**
   - Réseaux de Neurones LSTM pour capturer les séquences temporelles
   - XGBoost pour une précision supérieure
   - Ensemble de modèles (stacking)

3. **Validation Temporelle**
   - Walk-forward analysis (backtesting réaliste)
   - Cross-validation temporelle (Time Series Split)

4. **Mise en Production**
   - API REST pour prédictions en temps réel
   - Dashboard interactif avec Streamlit
   - Système d'alertes automatiques

---

## 8. RÉFÉRENCES ET OUTILS

**Dataset :** Kaggle - Crypto Volatility 2025  
**Algorithme :** Random Forest (Breiman, 2001)  
**Bibliothèques Python :**
- `pandas` : Manipulation de données
- `numpy` : Calculs numériques
- `scikit-learn` : Machine Learning
- `matplotlib` / `seaborn` : Visualisation

**Méthodologie :** CRISP-DM (Cross-Industry Standard Process for Data Mining)

---

