import networkx as nx
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import BOTH, X

# === ML imports ===
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# PARAMETERS & GLOBALS
# -----------------------------
NODE_RADIUS = 15

clf = None  # trained model
CLASS_NAMES = ["line", "star", "cycle", "complete", "tree", "general"]
FEATURE_NAMES = [
    "nodes",
    "edges",
    "density",
    "components",
    "cycles",
    "max_degree",
    "min_degree",
    "avg_degree",
    "diameter",
]

status_bar = None

LANGUAGES = {
    "fr": {
        "subtitle": "Exploration visuelle des graphes & apprentissage automatique",
        "theme": "Thème :",
        "language": "Langue :",
        "graphs_tab": "Graphes",
        "learning_tab": "Apprentissage IA",
        "manual_graphs": "Graphes manuels (plusieurs onglets possibles)",
        "new_graph": "Nouveau graphe",
        "close_graph": "Fermer ce graphe",
        "graph_tab": "Graphe {number}",
        "learn_header": "Comment l'IA apprend ?",
        "learn_intro": (
            "Dans cet onglet, on observe toutes les étapes de l'apprentissage supervisé :\n"
            "  1. Génération de nombreux graphes d'exemple.\n"
            "  2. Calcul des caractéristiques numériques pour chaque graphe.\n"
            "  3. Entraînement d'un arbre de décision.\n"
            "  4. Test sur de nouveaux graphes et calcul du taux de réussite.\n"
            "  5. Affichage des règles apprises par l'IA.\n\n"
            "Clique sur le bouton pour lancer tout le processus et lis les commentaires à droite."
        ),
        "start_learning": "Lancer l'apprentissage",
        "control_panel": "Panneau de contrôle",
        "instructions": (
            "• Clic gauche dans l'onglet Graphes : ajouter un sommet\n"
            "• Clic droit sur deux sommets : créer une arête\n"
            "• Shift + clic gauche sur un sommet : supprimer le sommet\n"
            "• « Nouveau graphe » : créer un nouvel onglet de graphe\n"
            "• « Apprentissage IA » : entraîner le modèle de classification\n"
        ),
        "clear_graph": "Effacer le graphe courant",
        "description_title": "Description du graphe :",
        "empty_graph": "Aucun graphe dessiné.",
        "special_one": "Cas particulier : 1 sommet (trivial)",
        "special_k2": "Cas particulier : K2 (à la fois ligne et complet)",
        "family_complete": "Famille mathématique (règles) : Graphe complet",
        "family_cycle": "Famille mathématique (règles) : Graphe en cycle",
        "family_star": "Famille mathématique (règles) : Graphe en étoile",
        "family_line": "Famille mathématique (règles) : Graphe en ligne (chemin)",
        "family_tree": "Famille mathématique (règles) : Graphe arborescent (arbre)",
        "family_general": "Famille mathématique (règles) : Graphe général",
        "stats_nodes_edges": "Sommets : {nodes}, Arêtes : {edges}",
        "stats_density": "Densité : {density}, Composantes : {components}, Cycles : {cycles}",
        "stats_degrees": "Degré min : {min_degree}, Degré max : {max_degree}, Degré moyen : {avg_degree}",
        "prediction": "Prédiction IA (apprentissage supervisé) : {label} ({confidence:.1f}% de confiance)",
        "prediction_untrained": "Prédiction IA : modèle non entraîné (onglet « Apprentissage IA »).",
        "prediction_few_nodes": "Prédiction IA : pas assez de sommets.",
        "ready": "Prêt. Choisissez un thème et créez un graphe.",
        "theme_changed": "Thème changé : {theme}",
        "language_changed": "Langue changée : Français",
        "new_graph_status": "Nouveau graphe créé (onglet {number}).",
        "graph_closed": "Graphe fermé, bascule vers un autre onglet.",
        "all_graphs_closed": "Tous les graphes ont été fermés.",
        "node_added": "Noeud ajouté (id {node}).",
        "node_selected": "Noeud {node} sélectionné – cliquez sur un autre sommet pour créer une arête.",
        "edge_added": "Arête ajoutée entre {first} et {second}.",
        "node_deleted": "Noeud {node} supprimé.",
        "graph_cleared": "Graphe effacé (onglet courant).",
        "learning_in_progress": "Apprentissage en cours...",
        "data_generation_status": "Génération du jeu de données...",
        "training_status": "Entraînement du modèle...",
        "trained_status": "IA entraînée – dessinez un graphe dans l'onglet Graphes.",
        "log_language_note": "Langue changée. Relancez l'apprentissage pour afficher le journal dans cette langue.",
        "type_line": "ligne",
        "type_star": "étoile",
        "type_cycle": "cycle",
        "type_complete": "complet",
        "type_tree": "arbre",
        "type_general": "général",
        "type_unknown": "inconnu",
    },
    "en": {
        "subtitle": "Visual graph exploration & machine learning",
        "theme": "Theme:",
        "language": "Language:",
        "graphs_tab": "Graphs",
        "learning_tab": "AI Learning",
        "manual_graphs": "Manual graphs (multiple tabs supported)",
        "new_graph": "New graph",
        "close_graph": "Close this graph",
        "graph_tab": "Graph {number}",
        "learn_header": "How does the AI learn?",
        "learn_intro": (
            "This tab shows every step of supervised learning:\n"
            "  1. Generate many example graphs.\n"
            "  2. Calculate numerical features for each graph.\n"
            "  3. Train a decision tree.\n"
            "  4. Test it on unseen graphs and calculate accuracy.\n"
            "  5. Display the rules learned by the AI.\n\n"
            "Click the button to run the whole process and read the comments on the right."
        ),
        "start_learning": "Start learning",
        "control_panel": "Control panel",
        "instructions": (
            "• Left-click in the Graphs tab: add a node\n"
            "• Right-click two nodes: create an edge\n"
            "• Shift + left-click a node: delete the node\n"
            "• ‘New graph’: create a new graph tab\n"
            "• ‘AI Learning’: train the classification model\n"
        ),
        "clear_graph": "Clear current graph",
        "description_title": "Graph description:",
        "empty_graph": "No graph drawn.",
        "special_one": "Special case: 1 node (trivial)",
        "special_k2": "Special case: K2 (both a path and complete)",
        "family_complete": "Mathematical family (rules): Complete graph",
        "family_cycle": "Mathematical family (rules): Cycle graph",
        "family_star": "Mathematical family (rules): Star graph",
        "family_line": "Mathematical family (rules): Path graph",
        "family_tree": "Mathematical family (rules): Tree graph",
        "family_general": "Mathematical family (rules): General graph",
        "stats_nodes_edges": "Nodes: {nodes}, Edges: {edges}",
        "stats_density": "Density: {density}, Components: {components}, Cycles: {cycles}",
        "stats_degrees": "Min degree: {min_degree}, Max degree: {max_degree}, Average degree: {avg_degree}",
        "prediction": "AI prediction (supervised learning): {label} ({confidence:.1f}% confidence)",
        "prediction_untrained": "AI prediction: model not trained (see the ‘AI Learning’ tab).",
        "prediction_few_nodes": "AI prediction: not enough nodes.",
        "ready": "Ready. Choose a theme and create a graph.",
        "theme_changed": "Theme changed: {theme}",
        "language_changed": "Language changed: English",
        "new_graph_status": "New graph created (tab {number}).",
        "graph_closed": "Graph closed; switched to another tab.",
        "all_graphs_closed": "All graphs have been closed.",
        "node_added": "Node added (id {node}).",
        "node_selected": "Node {node} selected – click another node to create an edge.",
        "edge_added": "Edge added between {first} and {second}.",
        "node_deleted": "Node {node} deleted.",
        "graph_cleared": "Graph cleared (current tab).",
        "learning_in_progress": "Learning in progress...",
        "data_generation_status": "Generating dataset...",
        "training_status": "Training model...",
        "trained_status": "AI trained – draw a graph in the Graphs tab.",
        "log_language_note": "Language changed. Run learning again to display the log in this language.",
        "type_line": "path",
        "type_star": "star",
        "type_cycle": "cycle",
        "type_complete": "complete",
        "type_tree": "tree",
        "type_general": "general",
        "type_unknown": "unknown",
    },
}

current_language = "fr"
next_graph_tab_number = 1


def tr(key, **kwargs):
    """Return a UI string in the currently selected language."""
    return LANGUAGES[current_language][key].format(**kwargs)


def graph_type_name(label):
    return tr(f"type_{label}") if f"type_{label}" in LANGUAGES[current_language] else label


def bi(fr, en):
    """Small helper for longer bilingual learning-log messages."""
    return en if current_language == "en" else fr

# per-canvas graph data so we can have multiple graph tabs
graph_data = {}         # canvas -> {"graph", "positions", "counter", "selected"}
frame_to_canvas = {}    # tab frame -> canvas
current_canvas = None   # canvas currently active

# -----------------------------
# STATUS BAR
# -----------------------------
def set_status(msg: str):
    global status_bar
    if status_bar is not None:
        status_bar.config(text=msg)


# -----------------------------
# GRAPH FEATURE FUNCTIONS
# -----------------------------
def graph_properties(G):
    """Calcule les propriétés d'un graphe G."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "components": 0,
            "cycles": 0,
            "diameter": 0,
            "max_degree": 0,
            "min_degree": 0,
            "avg_degree": 0.0,
        }

    degrees = [d for _, d in G.degree()]
    density = nx.density(G) if n > 1 else 0.0
    components = nx.number_connected_components(G)
    cycles = len(list(nx.cycle_basis(G)))
    diameter = nx.diameter(G) if nx.is_connected(G) and n > 1 else 0
    max_deg = max(degrees)
    min_deg = min(degrees)
    avg_deg = 2 * m / n  # degré moyen

    props = {
        "nodes": n,
        "edges": m,
        "density": round(density, 3),
        "components": components,
        "cycles": cycles,
        "diameter": diameter,
        "max_degree": max_deg,
        "min_degree": min_deg,
        "avg_degree": round(avg_deg, 3),
    }
    return props


def describe_graph(G):
    """Description par règles mathématiques (sans IA)."""
    if G.number_of_nodes() == 0:
        return tr("empty_graph")

    props = graph_properties(G)
    description = []

    n = props["nodes"]
    m = props["edges"]

    # Graphe en étoile
    if n == 1:
        description.append(tr("special_one"))
    elif n == 2 and m == 1:
        description.append(tr("special_k2"))

    # Graphe complet
    elif n > 1 and m == n * (n - 1) // 2:
        description.append(tr("family_complete"))
    # Cycle
    elif n > 2 and props["cycles"] == 1 and all(d == 2 for _, d in G.degree()):
        description.append(tr("family_cycle"))
    # Étoile (3 sommets est aussi un chemin, donc les étoiles commencent à 4 sommets)
    elif n >= 4 and sorted(d for _, d in G.degree()) == [1] * (n - 1) + [n - 1]:
        description.append(tr("family_star"))
    # Ligne / chemin
    elif (
        n > 1
        and nx.is_connected(G)
        and props["cycles"] == 0
        and all(d <= 2 for _, d in G.degree())
    ):
        description.append(tr("family_line"))
    # Arbre
    elif n > 0 and nx.is_tree(G):
        description.append(tr("family_tree"))
    else:
        description.append(tr("family_general"))

    # Infos numériques
    description.append(tr("stats_nodes_edges", nodes=props["nodes"], edges=props["edges"]))
    description.append(tr(
        "stats_density",
        density=props["density"],
        components=props["components"],
        cycles=props["cycles"],
    ))
    description.append(tr(
        "stats_degrees",
        min_degree=props["min_degree"],
        max_degree=props["max_degree"],
        avg_degree=props["avg_degree"],
    ))

    return "\n".join(description)


# -----------------------------
# ML PART
# -----------------------------
def extract_features(G):
    """Transforme un graphe en vecteur de caractéristiques numériques."""
    props = graph_properties(G)
    return [
        props["nodes"],
        props["edges"],
        props["density"],
        props["components"],
        props["cycles"],
        props["max_degree"],
        props["min_degree"],
        props["avg_degree"],
        props["diameter"],
    ]


def generate_training_data():
    """
    Génère un jeu de données :
    chaque exemple = (caractéristiques numériques, type de graphe).
    """
    X = []
    y = []

    # Graphes en ligne (chemins)
    for n in range(2, 15):
        G = nx.path_graph(n)
        X.append(extract_features(G))
        y.append("line")

    # Graphes en étoile
    for n in range(4, 15):
        G = nx.star_graph(n - 1)
        X.append(extract_features(G))
        y.append("star")

    # Cycles
    for n in range(4, 15):
        G = nx.cycle_graph(n)
        X.append(extract_features(G))
        y.append("cycle")

    # Graphes complets
    for n in range(3, 11):
        G = nx.complete_graph(n)
        X.append(extract_features(G))
        y.append("complete")

    # Arbres (arbres aléatoires)
    for n in range(3, 20):
        G = nx.random_labeled_tree(n)
        X.append(extract_features(G))
        y.append("tree")

    # Graphes généraux aléatoires
    for n in range(5, 20):
        for p in [0.2, 0.4, 0.6]:
            G = nx.gnp_random_graph(n, p)
            X.append(extract_features(G))
            y.append("general")

    return np.array(X), np.array(y)


def log(widget, msg):
    """Affiche un message à la fois dans la console et dans la zone de texte."""
    print(msg)
    if widget is not None:
        widget.insert("end", msg + "\n")
        widget.see("end")


def train_model(log_widget=None):
    """
    Entraîne l'arbre de décision et explique chaque étape
    dans l'onglet 'Apprentissage IA'.
    """
    global clf

    set_status(tr("data_generation_status"))
    log(log_widget, "══════════════════════════════════════════════")
    log(log_widget, bi("ÉTAPE 1 – Création du jeu de données", "STEP 1 – Create the dataset"))
    log(log_widget, bi(
        "On génère de nombreux graphes (ligne, étoile, cycle, complet, arbre, général).",
        "We generate many graphs (path, star, cycle, complete, tree, general).",
    ))
    log(log_widget, bi(
        "Pour chacun, on calcule : nombre de sommets, arêtes, densité, cycles, degrés, diamètre…",
        "For each graph, we calculate: nodes, edges, density, cycles, degrees, diameter…",
    ))
    X_all, y_all = generate_training_data()
    log(log_widget, bi(
        f"\nNombre total d'exemples générés : {len(X_all)}",
        f"\nTotal examples generated: {len(X_all)}",
    ))

    log(log_widget, bi("\nExemples de lignes de données :", "\nExample data rows:"))
    log(log_widget, bi(
        "(Chaque ligne = [caractéristiques numériques] -> type de graphe)",
        "(Each row = [numerical features] -> graph type)",
    ))
    for i in range(min(8, len(X_all))):
        log(log_widget, bi(
            f"Exemple {i+1} : {X_all[i].tolist()}  ->  {graph_type_name(y_all[i])}",
            f"Example {i+1}: {X_all[i].tolist()}  ->  {graph_type_name(y_all[i])}",
        ))

    # Séparation apprentissage / test
    log(log_widget, "\n══════════════════════════════════════════════")
    log(log_widget, bi("ÉTAPE 2 – Séparation en deux sous-ensembles", "STEP 2 – Split into two subsets"))
    log(log_widget, bi(
        "• Jeu d'entraînement : pour apprendre (environ 75 %)",
        "• Training set: used to learn (about 75%)",
    ))
    log(log_widget, bi(
        "• Jeu de test       : pour vérifier si l'IA a bien appris (environ 25 %)",
        "• Test set: used to check learning (about 25%)",
    ))
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=0, stratify=y_all
    )
    log(
        log_widget,
        bi(
            f"\nTaille du jeu d'entraînement : {len(X_train)}   /   Taille du jeu de test : {len(X_test)}",
            f"\nTraining set size: {len(X_train)}   /   Test set size: {len(X_test)}",
        ),
    )

    # Entraînement du modèle
    set_status(tr("training_status"))
    log(log_widget, "\n══════════════════════════════════════════════")
    log(log_widget, bi("ÉTAPE 3 – Entraînement de l'IA (arbre de décision)", "STEP 3 – Train the AI (decision tree)"))
    log(
        log_widget,
        bi(
            "On ajuste un arbre de décision qui apprend à prédire le type de graphe à partir des caractéristiques.",
            "We fit a decision tree that learns to predict graph type from the numerical features.",
        ),
    )
    clf = DecisionTreeClassifier(max_depth=6, random_state=0)
    clf.fit(X_train, y_train)
    log(log_widget, bi("\n→ Modèle entraîné sur le jeu d'entraînement.", "\n→ Model trained on the training set."))

    # Évaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    err = 1 - acc
    log(log_widget, "\n══════════════════════════════════════════════")
    log(log_widget, bi("ÉTAPE 4 – Évaluation de l'IA", "STEP 4 – Evaluate the AI"))
    log(
        log_widget,
        bi(
            "On montre au modèle des graphes qu'il n'a JAMAIS vus (jeu de test) et on regarde s'il devine bien.",
            "We show the model graphs it has NEVER seen (the test set) and check its predictions.",
        ),
    )
    log(
        log_widget,
        bi(f"\nTaux de réussite (accuracy) : {acc * 100:.2f} %", f"\nAccuracy: {acc * 100:.2f}%")
    )
    log(
        log_widget,
        bi(f"Taux d'erreur               : {err * 100:.2f} %", f"Error rate: {err * 100:.2f}%")
    )

    log(log_widget, bi("\nExemples de prédictions sur le jeu de test :", "\nExample predictions on the test set:"))
    log(log_widget, bi(
        "(Vrai type = étiquette humaine, Prédit = résultat de l'IA)",
        "(True type = human label, Predicted = AI result)",
    ))
    for i in range(min(12, len(X_test))):
        pred = clf.predict([X_test[i]])[0]
        log(
            log_widget,
            bi(
                f"Features={X_test[i].tolist()} | Vrai type={graph_type_name(y_test[i])} | Prédit={graph_type_name(pred)}",
                f"Features={X_test[i].tolist()} | True type={graph_type_name(y_test[i])} | Predicted={graph_type_name(pred)}",
            ),
        )

    # Règles apprises
    log(log_widget, "\n══════════════════════════════════════════════")
    log(log_widget, bi("ÉTAPE 5 – Règles apprises par l'arbre de décision", "STEP 5 – Rules learned by the decision tree"))
    log(
        log_widget,
        bi(
            "Ci-dessous, on voit l'arbre de décision sous forme de texte :",
            "Below is the decision tree in text form:",
        ),
    )
    log(
        log_widget,
        bi(
            "• chaque 'if' correspond à une question sur une caractéristique (ex : cycles <= 0.5 ?)",
            "• each 'if' asks a question about a feature (e.g. cycles <= 0.5?)",
        ),
    )
    log(
        log_widget,
        bi(
            "• les feuilles finales indiquent le type de graphe prédit (line / star / cycle / complete / tree / general).\n",
            "• final leaves show the predicted graph type (line / star / cycle / complete / tree / general).\n",
        ),
    )
    tree_text = export_text(clf, feature_names=FEATURE_NAMES)
    log(log_widget, tree_text)

    log(
        log_widget,
        bi(
            "\n>>> L'IA est maintenant entraînée. Vous pouvez retourner dans l'onglet « Graphes » et dessiner vos propres exemples.",
            "\n>>> The AI is now trained. You can return to the ‘Graphs’ tab and draw your own examples.",
        ),
    )
    set_status(tr("trained_status"))


def ml_predict_graph_type(G):
    """Utilise le modèle entraîné pour prédire le type de graphe."""
    if clf is None or G.number_of_nodes() < 2:
        return "unknown", 0.0
    features = np.array(extract_features(G)).reshape(1, -1)
    pred_label = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0].max()
    return pred_label, proba


# -----------------------------
# GUI SETUP
# -----------------------------
app = ttk.Window(
    title="Smart Graph AI Lab",
    themename="superhero",
    size=(1200, 720),
)

# ---- TOP BAR ----
topbar = ttk.Frame(app, padding=15, bootstyle="dark")
topbar.pack(side="top", fill=X)

title_label = ttk.Label(
    topbar,
    text="Smart Graph AI Lab",
    font=("Segoe UI", 20, "bold"),
)
title_label.pack(side="left")

subtitle_label = ttk.Label(
    topbar,
    text=tr("subtitle"),
    font=("Segoe UI", 10),
    foreground="#A0A0A0",
)
subtitle_label.pack(side="left", padx=(10, 0))

# Theme / language controls
top_controls = ttk.Frame(topbar, bootstyle="dark")
top_controls.pack(side="right")

theme_var = tk.StringVar(value="superhero")


def change_theme(event=None):
    theme = theme_var.get()
    app.style.theme_use(theme)
    set_status(tr("theme_changed", theme=theme))


theme_label = ttk.Label(top_controls, text=tr("theme"), font=("Segoe UI", 9))

theme_combo = ttk.Combobox(
    top_controls,
    textvariable=theme_var,
    values=["superhero", "flatly", "darkly", "morph"],
    width=10,
    state="readonly",
)
theme_combo.bind("<<ComboboxSelected>>", change_theme)

# Language switcher
language_var = tk.StringVar(value="Français")


def change_language(event=None):
    global current_language
    current_language = "en" if language_var.get() == "English" else "fr"
    update_ui_language()
    set_status(tr("language_changed"))


language_label = ttk.Label(top_controls, text=tr("language"), font=("Segoe UI", 9))

language_combo = ttk.Combobox(
    top_controls,
    textvariable=language_var,
    values=["Français", "English"],
    width=9,
    state="readonly",
)
language_combo.bind("<<ComboboxSelected>>", change_language)

language_label.pack(side="left", padx=(0, 5))
language_combo.pack(side="left", padx=(0, 15))
theme_label.pack(side="left", padx=(0, 5))
theme_combo.pack(side="left")

# ---- MAIN BODY ----
body = ttk.Frame(app, padding=10)
body.pack(fill=BOTH, expand=True)

# LEFT SIDE: sidebar
frame_left_outer = ttk.Frame(body, width=300)
frame_left_outer.pack(side="left", fill="y")
frame_left_outer.pack_propagate(False)

sidebar_card = ttk.Frame(frame_left_outer, padding=15, bootstyle="secondary")
sidebar_card.pack(fill="both", expand=True)

# RIGHT SIDE: notebook (Graphes / Apprentissage IA)
frame_right_outer = ttk.Frame(body)
frame_right_outer.pack(side="right", fill=BOTH, expand=True)

main_notebook = ttk.Notebook(frame_right_outer, bootstyle="info")
tab_graphs_root = ttk.Frame(main_notebook, padding=5)
tab_learn = ttk.Frame(main_notebook, padding=5)
main_notebook.add(tab_graphs_root, text=tr("graphs_tab"))
main_notebook.add(tab_learn, text=tr("learning_tab"))
main_notebook.pack(fill=BOTH, expand=True)

# ---- Graphes tab ----
graphs_toolbar = ttk.Frame(tab_graphs_root)
graphs_toolbar.pack(fill="x", pady=(0, 5))

graphs_label = ttk.Label(
    graphs_toolbar,
    text=tr("manual_graphs"),
    font=("Segoe UI", 11, "bold"),
)
graphs_label.pack(side="left")

graph_notebook = ttk.Notebook(tab_graphs_root, bootstyle="secondary")
graph_notebook.pack(fill=BOTH, expand=True)

# ---- Apprentissage IA tab ----
learn_header = ttk.Label(
    tab_learn,
    text=tr("learn_header"),
    font=("Segoe UI", 12, "bold"),
)
learn_header.pack(anchor="w", pady=(0, 5))

learn_intro = ttk.Label(
    tab_learn,
    text=tr("learn_intro"),
    justify="left",
    wraplength=700,
)
learn_intro.pack(anchor="w", pady=(0, 8))

learn_button = ttk.Button(
    tab_learn,
    text=tr("start_learning"),
    bootstyle="success-outline",
)
learn_button.pack(pady=5, anchor="w")

learn_frame = ttk.Frame(tab_learn, bootstyle="dark", padding=5)
learn_frame.pack(fill=BOTH, expand=True, pady=(5, 0))

learn_text = tk.Text(learn_frame, wrap="word", bg="#101820", fg="#E0E0E0", relief="flat")
learn_text.pack(side="left", fill=BOTH, expand=True)

learn_scroll = ttk.Scrollbar(learn_frame, orient="vertical", command=learn_text.yview)
learn_scroll.pack(side="right", fill="y")
learn_text.configure(yscrollcommand=learn_scroll.set)


def on_learn_button():
    learn_text.delete("1.0", "end")
    set_status(tr("learning_in_progress"))
    train_model(log_widget=learn_text)


learn_button.configure(command=on_learn_button)

# -----------------------------
# GRAPH TAB MANAGEMENT
# -----------------------------
def create_graph_tab():
    """Crée un nouvel onglet de graphe avec son propre canvas."""
    global current_canvas, next_graph_tab_number

    tab_index = next_graph_tab_number
    next_graph_tab_number += 1
    frame = ttk.Frame(graph_notebook, padding=5)
    graph_notebook.add(frame, text=tr("graph_tab", number=tab_index))
    graph_notebook.select(frame)

    container = ttk.Frame(frame, padding=5, bootstyle="dark")
    container.pack(fill=BOTH, expand=True)

    canvas = tk.Canvas(container, bg="#000000", highlightthickness=0)
    canvas.pack(fill=BOTH, expand=True)

    graph_data[canvas] = {
        "graph": nx.Graph(),
        "positions": {},
        "counter": 0,
        "selected": None,
        "tab_number": tab_index,
    }
    frame_to_canvas[frame] = canvas
    current_canvas = canvas

    canvas.bind("<Button-1>", lambda e, c=canvas: add_node(e, c))
    canvas.bind("<Button-3>", lambda e, c=canvas: select_node(e, c))
    canvas.bind("<Shift-Button-1>", lambda e, c=canvas: delete_node(e, c))

    update_description(canvas)
    set_status(tr("new_graph_status", number=tab_index))


def close_current_graph():
    """Ferme l'onglet de graphe courant."""
    global current_canvas

    if current_canvas is None:
        return

    frame_to_remove = None
    for frame, canvas in frame_to_canvas.items():
        if canvas is current_canvas:
            frame_to_remove = frame
            break

    if frame_to_remove is None:
        return

    graph_notebook.forget(frame_to_remove)
    del frame_to_canvas[frame_to_remove]
    del graph_data[current_canvas]

    if frame_to_canvas:
        any_frame = list(frame_to_canvas.keys())[0]
        graph_notebook.select(any_frame)
        canvas = frame_to_canvas[any_frame]
        current_canvas = canvas
        update_description(canvas)
        set_status(tr("graph_closed"))
    else:
        current_canvas = None
        label_description.config(text=f"{tr('description_title')}\n{tr('empty_graph')}")
        set_status(tr("all_graphs_closed"))


def on_graph_tab_changed(event):
    """Quand on change d'onglet de graphe."""
    global current_canvas
    nb = event.widget
    selected_tab = nb.select()
    if not selected_tab:
        return
    frame = nb.nametowidget(selected_tab)
    canvas = frame_to_canvas.get(frame)
    if canvas is None:
        return
    current_canvas = canvas
    update_description(canvas)


graph_notebook.bind("<<NotebookTabChanged>>", on_graph_tab_changed)

# toolbar buttons
btn_new_graph = ttk.Button(
    graphs_toolbar,
    text=tr("new_graph"),
    bootstyle="primary-outline",
    command=create_graph_tab,
)
btn_new_graph.pack(side="left", padx=(0, 5))

btn_close_graph = ttk.Button(
    graphs_toolbar,
    text=tr("close_graph"),
    bootstyle="danger-outline",
    command=close_current_graph,
)
btn_close_graph.pack(side="left")

# -----------------------------
# GRAPH INTERACTION FUNCTIONS
# -----------------------------
def draw_manual_graph(canvas):
    """Redessine le graphe sur un canvas donné."""
    data = graph_data[canvas]
    G = data["graph"]
    positions = data["positions"]
    selected = data["selected"]

    canvas.delete("all")
    # Arêtes
    for u, v in G.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        canvas.create_line(x1, y1, x2, y2, fill="#FFFFFF", width=2)
    # Sommets
    for node, (x, y) in positions.items():
        fill_color = "#5bc0de" if node != selected else "#f0ad4e"
        canvas.create_oval(
            x - NODE_RADIUS,
            y - NODE_RADIUS,
            x + NODE_RADIUS,
            y + NODE_RADIUS,
            fill=fill_color,
            outline="",
        )
        canvas.create_text(
            x, y, text=str(node), fill="black", font=("Segoe UI", 9, "bold")
        )


def add_node(event, canvas):
    """Clic gauche : ajoute un sommet sur ce canvas."""
    global current_canvas
    current_canvas = canvas
    data = graph_data[canvas]
    x, y = event.x, event.y
    node_id = data["counter"]
    data["positions"][node_id] = (x, y)
    data["graph"].add_node(node_id)
    data["counter"] += 1
    set_status(tr("node_added", node=node_id))
    draw_manual_graph(canvas)
    update_description(canvas)


def select_node(event, canvas):
    """Clic droit : sélectionne / relie deux sommets."""
    global current_canvas
    current_canvas = canvas
    data = graph_data[canvas]
    G = data["graph"]
    positions = data["positions"]
    selected = data["selected"]

    for node, (x, y) in positions.items():
        if (event.x - x) ** 2 + (event.y - y) ** 2 <= NODE_RADIUS ** 2:
            if selected is None:
                data["selected"] = node
                set_status(tr("node_selected", node=node))
                draw_manual_graph(canvas)
            else:
                if selected != node:
                    G.add_edge(selected, node)
                    set_status(tr("edge_added", first=selected, second=node))
                data["selected"] = None
                draw_manual_graph(canvas)
                update_description(canvas)
            break


def delete_node(event, canvas):
    """Shift + clic gauche : supprime un sommet."""
    global current_canvas
    current_canvas = canvas
    data = graph_data[canvas]
    G = data["graph"]
    positions = data["positions"]

    for node, (x, y) in list(positions.items()):
        if (event.x - x) ** 2 + (event.y - y) ** 2 <= NODE_RADIUS ** 2:
            if node in G:
                G.remove_node(node)
            del positions[node]

            if data["selected"] == node:
                data["selected"] = None

            set_status(tr("node_deleted", node=node))
            draw_manual_graph(canvas)
            update_description(canvas)
            return


def clear_graph():
    """Efface le graphe de l'onglet courant."""
    if current_canvas is None:
        return
    data = graph_data[current_canvas]
    data["graph"].clear()
    data["positions"].clear()
    data["counter"] = 0
    data["selected"] = None
    current_canvas.delete("all")
    label_description.config(text=f"{tr('description_title')}\n{tr('empty_graph')}")
    set_status(tr("graph_cleared"))


def update_description(canvas=None):
    """Met à jour le panneau de description en fonction du graphe courant."""
    if canvas is None:
        canvas = current_canvas
    if canvas is None:
        label_description.config(text=f"{tr('description_title')}\n{tr('empty_graph')}")
        return

    G = graph_data[canvas]["graph"]
    desc = describe_graph(G)

    ml_label, ml_conf = ml_predict_graph_type(G)

    if G.number_of_nodes() >= 2 and clf is not None and ml_label != "unknown":
        desc += "\n\n" + tr(
            "prediction",
            label=graph_type_name(ml_label),
            confidence=ml_conf * 100,
        )
    elif G.number_of_nodes() >= 2 and clf is None:
        desc += "\n\n" + tr("prediction_untrained")
    else:
        desc += "\n\n" + tr("prediction_few_nodes")

    label_description.config(text=f"{tr('description_title')}\n{desc}")


# -----------------------------
# SIDEBAR CONTENT
# -----------------------------
instr_title = ttk.Label(
    sidebar_card,
    text=tr("control_panel"),
    font=("Segoe UI", 13, "bold"),
)
instr_title.pack(anchor="w", pady=(0, 8))

instr_label = ttk.Label(
    sidebar_card,
    text=tr("instructions"),
    justify="left",
    wraplength=260,
)
instr_label.pack(anchor="w", pady=(0, 10))

btn_clear = ttk.Button(
    sidebar_card,
    text=tr("clear_graph"),
    bootstyle="danger",
    command=clear_graph,
)
btn_clear.pack(fill=X, pady=(0, 10))


def add_hover_cursor(widget):
    widget.bind("<Enter>", lambda e: widget.configure(cursor="hand2"))
    widget.bind("<Leave>", lambda e: widget.configure(cursor=""))


add_hover_cursor(btn_clear)
add_hover_cursor(learn_button)
add_hover_cursor(btn_new_graph)
add_hover_cursor(btn_close_graph)

ttk.Separator(sidebar_card, orient="horizontal").pack(fill=X, pady=8)

label_description = ttk.Label(
    sidebar_card,
    text=f"{tr('description_title')}\n{tr('empty_graph')}",
    font=("Segoe UI", 10),
    bootstyle="info",
    justify="left",
    anchor="w",
    wraplength=260,
)
label_description.pack(anchor="w", pady=5, fill=X)

# -----------------------------
# STATUS BAR
# -----------------------------
status_bar = ttk.Label(
    app,
    text=tr("ready"),
    bootstyle="secondary",
    anchor="w",
    padding=5,
)
status_bar.pack(side="bottom", fill=X)


def update_ui_language():
    """Refresh all visible UI text after a language change."""
    subtitle_label.config(text=tr("subtitle"))
    theme_label.config(text=tr("theme"))
    language_label.config(text=tr("language"))
    main_notebook.tab(tab_graphs_root, text=tr("graphs_tab"))
    main_notebook.tab(tab_learn, text=tr("learning_tab"))
    graphs_label.config(text=tr("manual_graphs"))
    learn_header.config(text=tr("learn_header"))
    learn_intro.config(text=tr("learn_intro"))
    learn_button.config(text=tr("start_learning"))
    btn_new_graph.config(text=tr("new_graph"))
    btn_close_graph.config(text=tr("close_graph"))
    instr_title.config(text=tr("control_panel"))
    instr_label.config(text=tr("instructions"))
    btn_clear.config(text=tr("clear_graph"))

    for frame, canvas in frame_to_canvas.items():
        number = graph_data[canvas]["tab_number"]
        graph_notebook.tab(frame, text=tr("graph_tab", number=number))

    if learn_text.get("1.0", "end-1c").strip():
        learn_text.delete("1.0", "end")
        learn_text.insert("end", tr("log_language_note"))

    update_description(current_canvas)

# -----------------------------
# INITIAL STATE
# -----------------------------
create_graph_tab()

app.mainloop()
