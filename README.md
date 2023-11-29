# Latent Topic Block Model

## Introduction
Ce dépôt GitHub contient une tentative d'implémentation du Latent Topic Block Model en Python, suivi d'une étude utilisant le jeu de données Amazon Fine Food Reviews.

## Prérequis
Avant de pouvoir exécuter les cellules du notebook, assurez-vous de suivre ces étapes préalables :

1. **Clonez le repo :** Utilisez la commande suivante pour cloner le dépôt sur votre machine locale.
   ```bash
   git clone https://github.com/vilhess/NSA.git
   cd NSA
   ```

2.  **Téléchargez le jeu de donnée :** Téléchargez le fichier CSV depuis le lien suivant et placez-le dans le même dossier que le notebook, en le renommant 'Reviews.csv'. [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

3. **Installez les dépendances :** Installez les dépendances Python en utilisant la commande suivante.
   ```bash
   pip install -r requirements.txt
   ```
4. **Exécutez le notebook :** Ouvrez le notebook dans un environnement compatible avec Jupyter et exécutez les cellules pour observer les résultats de l'implémentation du Latent Topic Block Model.
(Certaines matrices sont importées depuis des fichiers .csv pour éviter de les recalculer à chaque exécution du notebook.)