import numpy as np

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'value': np.mean(y)}

        best_feature, best_threshold, best_loss = None, None, float('inf')
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if sum(left_mask) < self.min_samples_split or sum(right_mask) < self.min_samples_split:
                    continue
                loss = self._split_loss(y, left_mask, right_mask)
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return {'value': np.mean(y)}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_tree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _split_loss(self, y, left_mask, right_mask):
        left_y, right_y = y[left_mask], y[right_mask]
        if len(left_y) == 0 or len(right_y) == 0:
            return float('inf')
        left_loss = np.var(left_y) * len(left_y)
        right_loss = np.var(right_y) * len(right_y)
        return (left_loss + right_loss) / len(y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if 'value' in node:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        y = np.array(y)
        if len(np.unique(y)) != 2:
            raise ValueError("Only binary classification is supported")
        y = np.where(y == np.unique(y)[0], 0, 1)  # Convert to 0/1
        n_samples = len(y)

        # Initial prediction: log-odds
        p = np.mean(y)
        self.initial_pred = np.log(p / (1 - p)) if p not in [0, 1] else 0
        current_pred = np.full(n_samples, self.initial_pred)

        for _ in range(self.n_estimators):
            # Compute probabilities
            p = 1 / (1 + np.exp(-current_pred))
            # Compute negative gradient (residuals)
            residuals = y - p
            # Fit a tree to the residuals
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            # Update predictions
            tree_pred = tree.predict(X)
            current_pred += self.learning_rate * tree_pred
            self.trees.append(tree)

    def predict_proba(self, X):
        if not self.trees:
            raise ValueError("Model not fitted")
        pred = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        proba = 1 / (1 + np.exp(-pred))
        return np.vstack((1 - proba, proba)).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)