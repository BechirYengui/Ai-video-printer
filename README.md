<div align="center">

<img src="tools/desktop/yvp.svg" width="96" alt="YVP" />

# YenguiVideoPrinter (YVP)

### AI Video Printer

**Des vidéos courtes complètes à partir d'un simple sujet, générées sur votre PC.**

Script, images, voix off, sous-titres et musique : YVP s'occupe de tout le montage.

![Linux](https://img.shields.io/badge/Linux-Ubuntu-E95420?logo=ubuntu&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-10%20%7C%2011-0078D4?logo=windows&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Licence](https://img.shields.io/badge/licence-MIT-green)
[![Dernier commit](https://img.shields.io/github/last-commit/BechirYengui/Ai-video-printer)](https://github.com/BechirYengui/Ai-video-printer/commits/main)

[Installation](#installation-sous-linux) · [Mise à jour](#mettre-à-jour) · [Dépannage](#dépannage) · [Signaler un problème](https://github.com/BechirYengui/Ai-video-printer/issues)

</div>

## Démarrage rapide

| Linux | Windows |
| :--- | :--- |
| `git clone https://github.com/BechirYengui/Ai-video-printer.git ~/YVP` | `git clone https://github.com/BechirYengui/Ai-video-printer.git C:\YVP` |
| `cd ~/YVP && ./install.sh` | Double-clic sur `C:\YVP\Installer-YVP.bat` |
| Icône **YenguiVideoPrinter** du dock | Icône **YenguiVideoPrinter** du Bureau |

Les détails, prérequis compris, sont dans les sections [Linux](#installation-sous-linux) et [Windows](#installation-sous-windows).

## Sommaire

1. [Démarrage rapide](#démarrage-rapide)
2. [Fonctionnalités](#fonctionnalités)
3. [Configuration requise](#configuration-requise)
4. [Installation sous Linux](#installation-sous-linux)
5. [Installation sous Windows](#installation-sous-windows)
6. [Lancer YVP](#lancer-yvp)
7. [Mettre à jour](#mettre-à-jour)
8. [Photo de profil LinkedIn](#photo-de-profil-linkedin)
9. [Réglages](#réglages)
10. [Dépannage](#dépannage)
11. [Désinstaller](#désinstaller)
12. [Structure du projet](#structure-du-projet)
13. [Contribuer](#contribuer)
14. [Crédits et licences](#crédits-et-licences)

## Fonctionnalités

| Module | Ce qu'il fait |
| :--- | :--- |
| **Vidéo courte** | À partir d'un sujet, l'IA écrit le script, choisit les images, pose la voix off, les sous-titres et la musique, puis monte la vidéo. |
| **Vos photos et vidéos** | Ajoutez vos propres fichiers : l'IA les regarde, écrit le script autour d'eux et place chacun au bon moment. |
| **Choix intelligent des visuels** | Un modèle de vision local vérifie que chaque plan correspond vraiment à la phrase prononcée. |
| **Vidéo explicative** | Une vidéo animée avec schémas, flèches et sous-titres mot à mot, appuyée sur des faits vérifiés. |
| **Studio caméra** | Filmez-vous devant le fond de votre choix, puis réutilisez la prise dans le montage. |
| **Photo de profil LinkedIn** | Vos propres photos deviennent des portraits professionnels, générés sur votre carte graphique. |

Points forts :

1. **Local d'abord.** Le modèle de langage tourne sur votre PC grâce à Ollama. Vos photos ne quittent jamais votre machine.
2. **Une seule fenêtre.** YVP démarre au clic sur son icône et s'arrête complètement quand on ferme sa fenêtre. Rien ne tourne en arrière-plan.
3. **15 langues d'interface**, dont le français, l'anglais et l'arabe.

## Configuration requise

| | Minimum | Recommandé |
| :--- | :--- | :--- |
| Système | Ubuntu 22.04 ou Windows 10 (64 bits) | Ubuntu 24.04 ou Windows 11 |
| Mémoire vive | 8 Go | 16 Go |
| Espace disque | 15 Go | 30 Go avec la photo LinkedIn |
| Carte graphique | Facultative | NVIDIA avec 8 Go de mémoire vidéo |
| Navigateur | Google Chrome (Linux), Edge ou Chrome (Windows) | |
| Connexion | Nécessaire pendant l'installation | |

La photo de profil LinkedIn exige une carte **NVIDIA avec au moins 8 Go de mémoire vidéo**. Tout le reste fonctionne sans carte graphique dédiée, simplement plus lentement.

## Installation sous Linux

**1. Installer les outils de base** (une seule fois) :

```bash
sudo apt install git curl zstd
```

Installez aussi [Google Chrome](https://www.google.com/chrome/) si ce n'est pas déjà fait : YVP s'ouvre dans une fenêtre Chrome dédiée.

**2. Récupérer le projet** :

```bash
git clone https://github.com/BechirYengui/Ai-video-printer.git ~/YVP
cd ~/YVP
```

**3. Lancer l'installation** :

```bash
./install.sh
```

L'installateur fait tout, dans cet ordre, et peut être relancé sans risque (chaque étape déjà faite est sautée) :

1. installe `uv`, le gestionnaire Python ;
2. installe Python 3.11 et toutes les dépendances dans `.venv` ;
3. crée `config.toml` à partir de `config.example.toml` ;
4. installe Ollama dans le dossier `ollama/` (aucun service système) ;
5. télécharge le modèle IA `qwen3:8b` (environ 5 Go) ;
6. ajoute l'icône YVP au menu des applications et au dock.

Comptez 20 à 40 minutes selon votre connexion.

## Installation sous Windows

**1. Installer Git pour Windows** (une seule fois) : téléchargez-le sur [git-scm.com](https://git-scm.com/download/win) et gardez les options par défaut.

**2. Récupérer le projet** : ouvrez *Windows PowerShell* et tapez :

```powershell
git clone https://github.com/BechirYengui/Ai-video-printer.git C:\YVP
```

Choisissez un dossier simple comme `C:\YVP`. Évitez OneDrive et le Bureau synchronisé.

**3. Lancer l'installation** : ouvrez le dossier `C:\YVP` et double-cliquez sur **`Installer-YVP.bat`**.

Si Windows affiche « Windows a protégé votre ordinateur », cliquez sur **Informations complémentaires** puis **Exécuter quand même**.

L'installateur fait les mêmes étapes que sous Linux : `uv`, Python et dépendances, `config.toml`, Ollama portable (environ 1,5 Go), modèle IA (environ 5 Go), puis les raccourcis sur le Bureau et dans le menu Démarrer. Attendez le message **« Installation terminée ! »**. Comptez 20 à 40 minutes.

Aucun autre logiciel n'est nécessaire : ni Python, ni Ollama, ni Visual Studio.

## Lancer YVP

| | Linux | Windows |
| :--- | :--- | :--- |
| Lancement | Icône **YenguiVideoPrinter** du dock ou du menu | Icône **YenguiVideoPrinter** du Bureau ou du menu Démarrer |
| Sans icône | `./start.sh` puis ouvrir http://127.0.0.1:8501 | |
| Arrêt | Fermer la fenêtre YVP | Fermer la fenêtre YVP |

Au clic, YVP démarre Ollama et l'interface, puis ouvre sa propre fenêtre. À la fermeture de cette fenêtre, tout s'arrête.

Astuce Windows : pour l'avoir dans la barre des tâches, faites un clic droit sur YenguiVideoPrinter dans le menu Démarrer, puis **Épingler à la barre des tâches**.

## Mettre à jour

Fermez d'abord la fenêtre YVP, puis :

| Linux | Windows |
| :--- | :--- |
| `cd ~/YVP` | `cd C:\YVP` |
| `git pull` | `git pull` |
| `./install.sh` | double-clic sur `Installer-YVP.bat` |

L'installateur ne refait que ce qui a changé, par exemple une nouvelle dépendance. Vos réglages (`config.toml`), vos vidéos et vos photos ne sont jamais touchés par une mise à jour.

## Photo de profil LinkedIn

Cette fonction transforme 5 à 20 photos de vous en portraits professionnels (tenue, fond, cadrage), sans rien envoyer sur internet.

**Activation**, une seule fois :

1. ouvrez la carte **Photo de profil LinkedIn** dans YVP ;
2. cliquez sur **Installer les outils** (environ 3 Go, plusieurs minutes) ;
3. lors de la première génération, les modèles se téléchargent (environ 8 Go, une seule fois).

**Utilisation** :

1. choisissez votre source : **Capture guidée** (recommandé, la caméra vous guide pose par pose) ou **Importer des photos** ;
2. choisissez la tenue, le fond, le nombre d'images et la qualité ;
3. cliquez sur **Générer**.

| Qualité | Taille | Durée indicative (GTX 1070) |
| :--- | :--- | :--- |
| Rapide | 768 × 768 | 4 min de chargement, puis 3 min par image |
| Haute | 1024 × 1024 | 4 min de chargement, puis 9 min par image |

Les photos sources sont effacées dès la fin de la génération. Les portraits se téléchargent au format JPEG carré attendu par LinkedIn.

Bon à savoir : la génération utilise surtout la carte graphique et environ 3 Go de mémoire vive. Si le PC manque de mémoire, seule la génération s'arrête : YVP reste ouvert et affiche un message. Sous Windows, fermer la fenêtre YVP arrête aussi une génération en cours.

## Réglages

Tous les réglages se font dans l'interface et sont enregistrés dans `config.toml`. Ce fichier contient vos clés d'API : il n'est jamais envoyé sur git.

| Réglage | Où |
| :--- | :--- |
| Modèle de langage (Ollama local, ou un service en ligne) | Paramètres de l'interface |
| Banque d'images (Pexels, Pixabay…) et clés d'API | Paramètres de l'interface |
| Voix off et langue | Carte 3 de l'interface |
| Autre modèle Ollama | Variable `MPT_OLLAMA_MODEL` avant l'installation, par exemple `MPT_OLLAMA_MODEL=qwen3:4b ./install.sh` |

Les vidéos produites sont rangées dans `storage/tasks`. Les enregistrements du studio caméra sont dans le dossier **Vidéos/YVP** de votre dossier personnel, sous Linux comme sous Windows.

## Dépannage

| Problème | Solution |
| :--- | :--- |
| YVP ne démarre pas | Consultez `logs/desktop.log` (Linux) ou `logs\desktop.err.log` (Windows), puis relancez l'installateur. |
| Le titre devient « Imprimante vidéo Yengui » | La traduction automatique du navigateur est désormais bloquée. Fermez puis rouvrez YVP. |
| YVP s'est fermé pendant une photo LinkedIn | Mémoire vive saturée : fermez les onglets lourds du navigateur et relancez. |
| « Mémoire vidéo insuffisante » | Fermez les jeux et logiciels qui utilisent la carte graphique, ou choisissez la qualité Rapide. |
| L'installation de la photo LinkedIn échoue | Le détail est dans `storage/headshot/install.log`. Relancez **Installer les outils**. |
| Le téléchargement du modèle IA coupe | Relancez simplement l'installateur : il reprend où il s'était arrêté. |
| Le port 8501 est occupé | YVP en choisit automatiquement un autre et ouvre la bonne adresse. |

## Désinstaller

| Linux | Windows |
| :--- | :--- |
| `tools/desktop/uninstall.sh` retire l'icône | Clic droit sur `tools\windows\uninstall.ps1`, puis **Exécuter avec PowerShell** |
| Supprimez ensuite le dossier `~/YVP` | Supprimez ensuite le dossier `C:\YVP` |

Les modèles de la photo LinkedIn sont dans le cache Hugging Face (`~/.cache/huggingface` sous Linux, `%USERPROFILE%\.cache\huggingface` sous Windows) et peuvent être supprimés à part.

## Structure du projet

```text
YVP/
├── install.sh              installation Linux
├── Installer-YVP.bat       installation Windows
├── start.sh                lancement manuel sous Linux
├── config.example.toml     modèle de configuration
├── app/                    moteur : script, visuels, voix, montage
│   └── services/           un fichier par fonction (headshot, explainer, curation…)
├── webui/                  interface (Streamlit) et traductions
├── resource/               polices, musiques, moteur des vidéos explicatives
├── tools/
│   ├── desktop/            lanceur et icône Linux
│   ├── windows/            installateur et lanceur Windows
│   └── headshot/           photo LinkedIn : installation et génération
├── storage/                vos vidéos et résultats (jamais envoyé sur git)
└── test/                   tests automatiques
```

## Contribuer

Les signalements de bugs et les suggestions sont les bienvenus dans les [issues](https://github.com/BechirYengui/Ai-video-printer/issues). Pour un signalement utile, joignez votre système (Linux ou Windows), votre carte graphique et les dernières lignes du fichier de log indiqué dans la section [Dépannage](#dépannage).

Pour travailler sur le code :

```bash
uv sync                # dépendances, outils de test compris
uv run pytest test     # tests automatiques
./start.sh             # lance l'interface sur http://127.0.0.1:8501
```

## Crédits et licences

YVP est construit sur [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) de harry0703, sous licence MIT. La documentation d'origine reste disponible : [中文](README-zh.md), [English](README-en.md), [日本語](README-ja.md).

| Composant | Rôle | Licence |
| :--- | :--- | :--- |
| [Ollama](https://ollama.com) et [Qwen3](https://github.com/QwenLM/Qwen3) | Modèle de langage local | MIT, Apache 2.0 |
| [PhotoMaker V2](https://github.com/TencentARC/PhotoMaker) | Ressemblance des portraits | Apache 2.0 |
| [RealVisXL V4](https://huggingface.co/SG161222/RealVisXL_V4.0) | Génération des portraits (SDXL) | OpenRAIL++ |
| [InsightFace buffalo_l](https://github.com/deepinsight/insightface) | Détection et identité du visage | Usage non commercial uniquement |
| [Tabler Icons](https://tabler.io/icons) | Icônes des vidéos explicatives | MIT |

Les modèles de visage InsightFace sont réservés à un usage personnel et non commercial : la photo de profil LinkedIn est prévue pour votre propre profil.

Le code de YVP est distribué sous [licence MIT](LICENSE).
