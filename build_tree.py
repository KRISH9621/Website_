import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. LOAD THE REAL DATA
df = pd.read_csv("heart.csv")
X = df.drop('target', axis=1)
y = df['target']
feature_names = X.columns.tolist()
class_names = ["Healthy", "Heart Disease"]

# 2. TRAIN A RANDOM FOREST (We use 3 trees to keep it lightweight)
clf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
clf.fit(X, y)

# 3. EXTRACT THE MATH FOR MULTIPLE TREES
forest_data = []

for index, tree_model in enumerate(clf.estimators_):
    tree_ = tree_model.tree_
    nodes = []
    links = []

    def traverse_tree(node_id, depth):
        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
        
        current_id = int(node_id)
        impurity = float(round(tree_.impurity[node_id], 3))
        samples = int(tree_.n_node_samples[node_id])
        
        value = tree_.value[node_id][0]
        class_idx = int(np.argmax(value))
        predicted_class = class_names[class_idx]

        if is_leaf:
            nodes.append({
                "id": f"T{index}_{current_id}", # Unique ID for each tree's nodes
                "type": "Leaf",
                "rule": "Prediction",
                "samples": samples,
                "impurity": impurity,
                "class": predicted_class,
                "depth": depth
            })
        else:
            feature_name = feature_names[tree_.feature[node_id]]
            threshold = float(round(tree_.threshold[node_id], 1))
            rule = f"{feature_name} <= {threshold}"
            
            nodes.append({
                "id": f"T{index}_{current_id}",
                "type": "Decision",
                "rule": rule,
                "samples": samples,
                "impurity": impurity,
                "class": None,
                "depth": depth
            })

            left_child = int(tree_.children_left[node_id])
            links.append({"source": f"T{index}_{current_id}", "target": f"T{index}_{left_child}", "condition": "True"})
            traverse_tree(left_child, depth + 1)

            right_child = int(tree_.children_right[node_id])
            links.append({"source": f"T{index}_{current_id}", "target": f"T{index}_{right_child}", "condition": "False"})
            traverse_tree(right_child, depth + 1)

    traverse_tree(0, 0)
    
    # Add this specific tree to our forest
    forest_data.append({
        "tree_name": f"Expert Tree {index + 1}",
        "nodes": nodes,
        "links": links
    })

# 4. SAVE TO JSON
with open("random_forest.json", "w") as f:
    json.dump(forest_data, f, indent=4)

print("Random Forest successfully exported to random_forest.json!")