# The North Face E-Commerce -- Recommandation et Topic Modeling

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=fff)](#)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=fff)](#)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=fff)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

---

## About

The North Face souhaite **booster les ventes en ligne** de son site e-commerce en exploitant les descriptions de son catalogue produit. Deux leviers identifiés par le département marketing :

- Système de recommandation -- déployer une section "Vous aimerez aussi..." sur chaque fiche produit pour augmenter le taux de conversion et le panier moyen.
- Restructuration du catalogue -- identifier des thématiques latentes dans les descriptions afin de proposer de nouvelles catégories de navigation plus pertinentes.

Le projet combine du **NLP** (tokenisation, lemmatisation, TF-IDF) et de l'**apprentissage non supervisé** (clustering DBSCAN, topic modeling LSA) pour répondre à ces deux objectifs.

---

## Dataset

| Propriété | Valeur |
|-----------|--------|
| Source | [Kaggle -- Product Item Data](https://www.kaggle.com/cclark/product-item-data?select=sample-data.csv) |
| Nombre de produits | 500 |
| Colonnes | `id`, `description` |
| Langue | Anglais |

Le fichier `sample-data.csv` est placé dans `data/raw/`.
Le dossier `data/` est gitignoré : télécharger le CSV depuis Kaggle et le placer manuellement dans `data/raw/`.

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/athanormark/THE_NORTH_FACE-_-BLOC-3_JEDHA_FORMATION.git
cd THE_NORTH_FACE-_-BLOC-3_JEDHA_FORMATION

# 2. Créer un environnement virtuel
conda create --name the_north_face python=3.13
conda activate the_north_face

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le modèle de langue spaCy
python -m spacy download en_core_web_sm

# 5. Lancer le notebook
jupyter notebook notebooks/01_EDA_and_Modeling.ipynb
```

---

## Pipeline

### 1. Preprocessing NLP
- Tokenisation, suppression des stop words, lemmatisation via **spaCy** (`en_core_web_sm`).
- Vectorisation **TF-IDF** (`max_features=1000`) produisant une matrice (500 x 1000).

### 2. Clustering DBSCAN
- Métrique : distance cosinus.
- Deux itérations de tuning sur `eps` (`min_samples=3`), justifiées par la courbe k-distance :

| eps | Clusters | Outliers | Commentaire |
|-----|----------|----------|-------------|
| 0.2 | 19 | 401 (80 %) | Trop restrictif |
| **0.63** | **17** | **20 (4 %)** | Cible atteinte (10-20 clusters) |

Le choix de `eps=0.63` est confirmé par la courbe k-distance (coude dans la zone 0.6-0.65). Silhouette Score : 0.086 (positif, cohérent avec du texte TF-IDF en haute dimension).

Principaux clusters identifiés :
- **Cluster 1** (295 produits) -- Vêtements éco-responsables : `recyclable`, `organic`, `cotton`.
- **Cluster 0** (61 produits) -- Baselayers techniques : `gladiodor`, `odor`, `natural`.
- **Cluster 2** (28 produits) -- Sacs et accessoires : `pocket`, `shoulder`, `strap`.

### 3. Système de recommandation
La fonction `find_similar_items(item_id)` retourne 5 produits du même cluster.
Interface interactive via `input()`.

### 4. Topic Modeling (LSA / TruncatedSVD)
- `TruncatedSVD(n_components=10)` -- Variance expliquée : 25.1 %.
- Topic dominant extrait par document.
- Visualisation par WordClouds.

---

## Résultats

### Clustering DBSCAN -- Regroupement des produits

| Configuration | Clusters | Outliers | Produits classés |
|---------------|----------|----------|------------------|
| eps=0.2 (baseline) | 19 | 401 (80%) | 20% |
| **eps=0.63 (retenu)** | **17** | **20 (4%)** | **96%** |

Silhouette Score : **0.086** (positif, cohérent pour du TF-IDF haute dimension). Les 3 clusters principaux : éco-responsable (295 produits), outdoor technique (61 produits), équipement DWR (28 produits).

### Topic Modeling LSA -- Thématiques du catalogue

| Topic | Mots-clés | Interprétation | Produits |
|-------|-----------|----------------|----------|
| 1 | `organic`, `cotton`, `recyclable` | Éco-responsable | 301 |
| 2 | `shirt`, `ringspun`, `phthalate`, `ink` | Sérigraphie | 33 |
| 3 | `merino`, `odor`, `gladiodor`, `wool` | Laine technique | 38 |
| 4 | `button`, `canvas`, `jean`, `welt` | Casual coton | 21 |
| 5 | `merino`, `wool`, `wash`, `chlorine` | Entretien laine | 17 |
| 6 | `sun`, `upf`, `nylon`, `protection` | Protection solaire | 7 |
| 7 | `strap`, `pocket`, `mesh`, `compartment` | Sacs et rangement | 28 |
| 8 | `spandex`, `tencel`, `bra`, `dress` | Vêtements femme | 37 |
| 9 | `photo`, `poster`, `outdoor`, `retail` | Marketing/retail | 8 |
| 10 | `sun`, `upf`, `collar`, `rashguard` | Protection UV | 10 |

---

## Conclusion

Le projet répond aux deux objectifs posés par The North Face :

Côté recommandation, DBSCAN avec distance cosinus identifie 17 clusters thématiques (4 % d'outliers). Le cluster dominant regroupe les vêtements éco-responsables (295 produits sur 500). La fonction `find_similar_items(item_id)` retourne 5 produits du même cluster, exploitable en production pour un bloc "Vous aimerez aussi".

Côté restructuration du catalogue, LSA (10 composantes, 25.1 % de variance expliquée) extrait des topics interprétables : éco-responsable, sérigraphie, laine technique, protection UV, sacs et rangement. Ces thématiques peuvent alimenter de nouvelles catégories de navigation sur le site e-commerce.

Recommandations marketing : mettre en avant la gamme éco-responsable qui représente 60 % du catalogue (Topic 1), créer une catégorie dédiée Protection Solaire (Topics 6+10), et valoriser la laine technique Merino (Topics 3+5) comme argument premium.

Limites : le dataset est restreint (500 produits) et la variance expliquée par LSA est faible (normal en NLP haute dimension). Les recommandations restent intra-cluster (pas de similarité fine) et le corpus est monolingue (anglais).

---

## Structure du projet

```text
TheNorthFace_Recommendation/
|
|-- data/
|   |-- raw/                # Données sources (sample-data.csv)
|   +-- processed/          # Matrices sauvegardées (optionnel)
|
|-- notebooks/
|   +-- 01_EDA_and_Modeling.ipynb   # Notebook principal
|
|-- assets/
|   +-- images/             # Captures, WordClouds
|
|-- requirements.txt        # Dépendances Python
|-- .gitignore
+-- README.md
```

---

## Auteur

Athanor SAVOUILLAN · [GitHub](https://github.com/athanormark)
