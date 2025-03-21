# Jeu Snake avec Intelligence Artificielle

# Description

Ce projet est une version du jeu classique Snake, où le joueur contrôle un serpent qui doit manger de la nourriture tout en évitant les murs, son propre corps et d'autres obstacles. Le jeu inclut une **intelligence artificielle (IA)** qui tente de jouer contre l'humain, mais elle est encore en développement et peut rencontrer des problèmes.

# Fonctionnalités :
- Le joueur contrôle un serpent qui mange des objets pour grandir.
- L'IA essaie de jouer contre l'humain, mais elle a des difficultés à éviter son propre corps et les obstacles.
- Le jeu propose des contrôles simples pour déplacer le serpent (haut, bas, gauche, droite).
- L'objectif est de survivre le plus longtemps possible sans entrer en collision.

# État actuel :
- Le jeu fonctionne avec une IA de base, mais elle peut se cogner contre son propre corps ou des obstacles présents sur le terrain.
- L'IA prend des décisions pour déplacer le serpent, mais elle rencontre encore des difficultés à éviter les obstacles et gérer ses mouvements efficacement.
- Des améliorations futures incluent des algorithmes de recherche de chemin (comme A*) et des stratégies de navigation plus avancées pour rendre l'IA plus compétitive et éviter les collisions.

# Installation

1. Clonez ce repository :
   ```bash
   git clone [https://github.com/elhadji4/Snack_game_AI-.git]

    Installez les dépendances requises :

pip install -r requirements.txt

Lancez le jeu :

    python main.py

Développement futur

    Amélioration de l'IA pour éviter les collisions avec son propre corps et les obstacles.
    Implémentation d'algorithmes de recherche de chemin (A*, Dijkstra) pour optimiser les déplacements de l'IA et réduire les risques de collision.
    Ajout de niveaux ou de modes de jeu supplémentaires pour enrichir l'expérience utilisateur.

Contribuer

Si vous souhaitez contribuer à ce projet, vous pouvez soumettre une pull request avec des améliorations ou des corrections de bugs.
Auteurs

    Nahim Haadji - Développeur principal


Cette version précise que l'IA se cogne parfois contre des obstacles, en plus de son propre corps
