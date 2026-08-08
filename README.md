# Smart Graph AI Lab

An interactive graph theory and machine learning application built with Python, NetworkX and scikit-learn.

Developed as part of **MATh.en.JEANS**, the project allows users to draw graphs, analyze their mathematical properties, train a Decision Tree classifier and observe how the model classifies new graph structures.

## Screenshots
<img width="2556" height="1345" alt="image" src="https://github.com/user-attachments/assets/142a9e2f-66c9-4a5e-81cb-ec61cc3d96c9" />
<img width="1650" height="1027" alt="image" src="https://github.com/user-attachments/assets/8b545497-7b33-49cd-a9ca-e4fa843f6c53" />
<img width="1222" height="982" alt="image" src="https://github.com/user-attachments/assets/bb543f26-cf55-4e62-89cc-78e66b141199" />
<img width="1838" height="1083" alt="image" src="https://github.com/user-attachments/assets/5f19e663-e893-47b2-a226-0960bd59aa18" />

## Key Features

- Interactive graph creation
- Mathematical graph analysis
- Rule-based graph classification
- Machine learning classification
- Automatic synthetic dataset generation
- Decision Tree training and evaluation
- Model confidence scores
- Interpretable learned decision rules
- English and French interface


## How It Works

Smart Graph AI Lab is an interactive graph-theory application that combines **mathematical graph analysis** with **supervised machine learning**.

The user can manually construct graphs, inspect their mathematical properties, train a machine-learning model, and then let the trained model classify newly drawn graphs.

### 1. Creating a Graph

Graphs are represented internally using **NetworkX**.

The graphical interface allows the user to:

* Left-click to create a node
* Right-click two nodes to connect them with an edge
* Shift + left-click a node to remove it
* Create multiple independent graph tabs
* Clear or close graphs

Each graph tab maintains its own NetworkX graph, node positions, node counter, and currently selected node.

This allows several different graphs to be created and analyzed at the same time.

---

### 2. Mathematical Graph Analysis

Every time the graph changes, Smart Graph AI Lab calculates a set of mathematical properties describing its structure.

For a graph (G), the application calculates:

* Number of nodes
* Number of edges
* Graph density
* Number of connected components
* Number of independent cycles
* Maximum node degree
* Minimum node degree
* Average node degree
* Diameter of the graph

For example, a graph can internally be represented as:

```text
Nodes: 6
Edges: 5
Density: 0.333
Components: 1
Cycles: 0
Max degree: 3
Min degree: 1
Average degree: 1.667
Diameter: 4
```

These measurements describe the graph numerically and are later used as inputs for the machine-learning model.

The average node degree is calculated using:

```text
2 × number_of_edges
───────────────────
  number_of_nodes
```

---

### 3. Rule-Based Classification

Before machine learning is involved, the application can also identify graph families using mathematical rules.

It checks whether the graph corresponds to one of several common structures:

* **Complete graph**
* **Cycle graph**
* **Star graph**
* **Path graph**
* **Tree**
* **General graph**

For example, a complete graph with (n) nodes is detected when the number of edges satisfies:

```text
m = n(n - 1) / 2
```

A cycle is detected when the graph contains one cycle and every node has degree 2.

A path is identified as a connected graph with no cycles and no node having degree greater than 2.

This deterministic classification is separate from the machine-learning prediction and provides a mathematical reference for the user.

---

### 4. Feature Extraction

Machine-learning algorithms cannot directly process the visual representation of a graph.

Smart Graph AI therefore converts each NetworkX graph into a numerical **feature vector**.

The feature vector contains:

```text
[
    nodes,
    edges,
    density,
    components,
    cycles,
    max_degree,
    min_degree,
    average_degree,
    diameter
]
```

For example:

```text
[5, 4, 0.4, 1, 0, 2, 1, 1.6, 4]
```

This numerical representation becomes the input used by the machine-learning classifier.

---

### 5. Automatic Training Dataset Generation

Instead of relying on an external dataset, the application generates its own labeled graph dataset using NetworkX.

Training examples are automatically created for six graph categories.

#### Path graphs

Path graphs containing between 2 and 14 nodes are generated.

#### Star graphs

Star graphs containing between 4 and 14 nodes are generated.

#### Cycle graphs

Cycle graphs containing between 4 and 14 nodes are generated.

#### Complete graphs

Complete graphs containing between 3 and 10 nodes are generated.

#### Trees

Random labeled trees containing between 3 and 19 nodes are generated.

#### General graphs

Random Erdős–Rényi graphs are generated with:

```text
5–19 nodes
```

and three different edge probabilities:

```text
p = 0.2
p = 0.4
p = 0.6
```

In total, the program generates **105 labeled graph examples**.

Each generated graph is transformed into its nine numerical features and paired with its correct graph category.

Conceptually, the dataset looks like:

```text
[features]                         → class

[5, 4, 0.4, 1, 0, 2, 1, 1.6, 4] → line
[6, 5, ...]                       → star
[8, 8, ...]                       → cycle
[6, 15, ...]                      → complete
...
```

---

### 6. Training and Test Split

The generated dataset is divided into two subsets:

```text
≈ 75% → Training data
≈ 25% → Test data
```

The split uses a fixed random state so that experiments remain reproducible.

Stratification is also used to preserve the distribution of the six graph categories across the training and test sets.

The training dataset is used by the AI to learn patterns.

The test dataset contains graphs that were not used during training and is used to evaluate whether the classifier has learned to recognize graph structures.

---

### 7. Decision Tree Machine Learning

Smart Graph AI Lab uses a **Decision Tree Classifier** from scikit-learn.

The model receives the nine numerical graph properties as inputs and learns which combinations of those properties correspond to each graph family.

The classifier is configured with:

```python
DecisionTreeClassifier(
    max_depth=6,
    random_state=0
)
```

Limiting the tree depth keeps the learned model relatively small and makes its decisions easier to inspect.

During training, the decision tree learns rules similar to:

```text
Is the number of cycles <= 0.5?
│
├── Yes
│   └── Check node degrees...
│
└── No
    └── Check density...
```

The exact rules are not manually programmed.

Instead, they are learned from the generated training examples.

---

### 8. Model Evaluation

After training, the classifier predicts the categories of graphs in the test dataset.

The predicted labels are compared with the true graph types.

The program then calculates:

```text
Accuracy = Correct predictions / Total predictions
```

and:

```text
Error rate = 1 - Accuracy
```

Several individual predictions are also displayed so the user can compare:

```text
True graph type
vs.
AI predicted graph type
```

This makes the machine-learning process visible rather than treating the classifier as a black box.

---

### 9. Inspecting What the AI Learned

One of the educational features of Smart Graph AI Lab is that it displays the rules learned by the decision tree.

Using scikit-learn's decision-tree export functionality, the trained model is converted into a readable text representation.

The user can therefore inspect which properties the model relies on when distinguishing between different graph structures.

For example, the tree might evaluate properties such as:

```text
cycles
density
maximum degree
number of edges
diameter
```

before reaching a final classification.

This helps demonstrate how supervised machine learning transforms training examples into decision rules.

---

### 10. Classifying a User-Created Graph

Once training is complete, the user can return to the graph editor and draw a new graph.

The complete prediction pipeline is:

```text
User draws a graph
        │
        ▼
NetworkX graph representation
        │
        ▼
Calculate graph properties
        │
        ▼
Extract 9 numerical features
        │
        ▼
Decision Tree Classifier
        │
        ▼
Predicted graph category
        │
        ▼
Prediction + confidence displayed
```

The model returns both:

* The predicted graph category
* A confidence score derived from the classifier's predicted probabilities

For example:

```text
AI prediction: star
Confidence: 96.4%
```

The user can then compare the AI prediction with the graph's rule-based mathematical classification.

---

## AI Learning Pipeline

```text
                 ┌──────────────────────┐
                 │ Generate Graphs      │
                 │ using NetworkX       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Calculate Graph      │
                 │ Properties           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Extract 9 Numerical  │
                 │ Features             │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Training / Test      │
                 │ Split                │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Decision Tree        │
                 │ Training             │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Test Predictions     │
                 │ + Accuracy           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Classify User-Drawn  │
                 │ Graphs               │
                 └──────────────────────┘
```

## Technologies

* **Python** — core application logic
* **NetworkX** — graph representation, generation and mathematical analysis
* **NumPy** — numerical dataset representation
* **scikit-learn** — decision-tree training, prediction and evaluation
* **Tkinter** — graphical interface and graph canvas
* **ttkbootstrap** — modern styling and themes
* **PyInstaller** — standalone Windows executable distribution


## MATh.en.JEANS

This project was developed as part of MATh.en.JEANS, an educational mathematics research initiative.

The objective of the project was to explore graph theory interactively and investigate how machine learning can learn to distinguish between different mathematical graph structures.

## Author

**Demir Dogan**

Student project developed for MATh.en.JEANS.

## License

This project is intended for educational and portfolio purposes.

See the LICENSE file for additional information.


## Run locally

pip install -r requirements.txt
python src/main.py
