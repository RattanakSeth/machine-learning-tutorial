class Node():
    """Base class for a tree node
    
    We will use a single class to have the options to build a Decision Tree,
    a Random Forest and Adaboost.
    
    Attributes:
        X: A torch tensor for the data.
        y: A torch tensor for the labels.
        is_categorical_list: A list of boolean, 1 if the features is categorical,
            0 otherwise.
        max_depth: A positive integer for the depth of the tree.
        columns: A list of the name of each column.
        depth: A positive integer for the depth of the node.
        nb_features: A positive integer for selecting a number of random
            features for the split. Only used when training a Random Forest.
        weights: A torch tensor for computing the weighted gini index.
            Only used when training Adaboost.
        col_perm: A torch tensor of the permuted column. Only for Random Forests.
        X_full: A torch tensor that is keeping the whole data in case we shuffle
            the input for training a Random Forest.
        col_full: A list that keeps the whole feature names. Only for Random Forests.
        is_catfull: A list that keeps the whole is_categorical_list if training a
            Random Forest.
        size: A positive integer for the number of samples.
        nb_features: A positive integer for the number of features
        cutoff: A float for the splitting value of the node.
        col: A positive integer as an index for which column to apply the condition
            for the split.
        left: A Node class for the left child node.
        right: A Node class for the right child node.
        
    """
    def __init__(self, X, y, is_categorical, max_depth, columns, depth=0, nb_features=0, weights=None):
        """Init function of the class
            
        This function builds the whole tree, recursively creating each child node down
        to the leaves.
        
        Args:
            Described in the attribute section of the class
        """
        # If training random forest 
        if nb_features != 0:
            self.col_perm = torch.randperm(X.shape[1])[:nb_features]
            # We have to keep the permutated data as well as the full data
            # because we have to pass the full data to the child nodes
            self.X = X[:, self.col_perm]
            self.X_full = X
            self.columns = columns[self.col_perm]
            self.col_full = columns
            self.is_categorical_list = is_categorical[self.col_perm]
            self.is_catfull = is_categorical
        else:
            # Regular training of the decision tree
            self.X = X
            self.columns = columns
            self.is_categorical_list = is_categorical
        # Weights are used to compute a weighted gini index
        # Only when training AdaBoost
        self.weights = weights
        self.y = y
        self.size = float(X.shape[0])
        # Prediction if the node will turn into a leaf, or split value
        self.cutoff = None
        # Column to check for splitting the data 
        self.col = None
        # Child nodes
        self.left = None
        self.right = None
        # Whether or not the split value is categorical
        self.depth = depth
        self.nb_features = nb_features
        # If the node contains only one label, it is set as a leaf
        if self.is_pure() or depth == max_depth or self.X.shape[0] == 1:
            # Select the predominant categorical value and set it as the prediction
            self.make_leaf()
            return
            
        # Computing the gini index on the population before the split
        gini_n = self.gini(y, self.weights)
        params = self.find_best_split()
        
        gini_s = params[0]
        # If no improvement, make the node as a leaf
        if gini_s >= gini_n:
            self.make_leaf()
        else:
            self.make_split(params, max_depth)

    def gini(self, y, weights=None):
        """Computes the gini index
        
        Args:
            y: A torch tensor for the labels of the data.
            weights: If training a Random Forest, the weights associated
                with each sample.
        """
        if weights is None:
            # Regular gini index
            pi = y.bincount() / float(y.shape[0])
        else:
            # Weighted gini index
            pi = self.get_weighted_population(y, weights)
        return 1 - pi.pow(2).sum()
    
    # For weighted gini index:
    def get_weighted_population(self, y, weights):
        """Computes the weighted gini index
        
        Instead of counting the samples for each class
        we count the weights of each class and divide by the sum of the weights
        
        Args:
            y: A torch tensor for the labels of the data.
            weights: If training a Random Forest, the weights associated
        """
        pi = torch.zeros((2,), dtype=torch.float32)
        idx_0 = torch.where(y == 0)[0]
        idx_1 = torch.where(y == 1)[0]
        pi[0] = weights[idx_0].sum()
        pi[1] = weights[idx_1].sum()
        return pi / weights.sum()
        
    def clean_attributes(self):
        """Cleans variables that are not usefull to predict"""
        del(self.X)
        del(self.y)
        del(self.weights)
        del(self.size)
        if self.nb_features != 0:
            del(self.X_full)
            del(self.col_full)
            del(self.is_catfull)
            del(self.colperm)
        del(self.nb_features)
    
    def is_pure(self):
        """Checks if the node is pure
        
        The node is pure if there is only one label in the population
        """
        return len(self.y.unique()) == 1
    
    def get_label(self):
        """Returns the most present label as a prediction if the node turns to a leaf"""
        if self.weights is None:
            return self.y.bincount().argmax().item()
        else:
            return self.get_weighted_population(self.y, self.weights).argmax().item()
    
    def make_leaf(self):
        """Makes the node a leaf"""
        self.cutoff = self.get_label()
        self.clean_attributes()
    
    def make_split(self, params, max_depth):
        """Performs the split
        
        Args:
            params: See find_best_split function
            max_depht: A positive integer for the maximum of the tree.
        """
        self.col = params[1]
        self.cutoff = params[2]
        # Save the categorical boolean of the selected column for the predict method
        self.var_categorical = self.is_categorical_list[self.col].item()
        # Recursively split by creating two instances of the Node class using the two groups
        if self.nb_features != 0:
            cols = self.col_full
            categorical_list = self.is_catfull
        else:
            cols = self.columns
            categorical_list = self.is_categorical_list
        # Creating child nodes based on the best params
        # If training Random Forest, we pass nb_features
        # If training AdaBoost, we pass the weights
        self.left = Node(X=params[3][0],
                         y=params[3][1],
                         is_categorical=categorical_list,
                         max_depth=max_depth,
                         columns=cols,
                         depth=self.depth + 1,
                         nb_features=self.nb_features,
                         weights=params[5])
        self.right = Node(X=params[4][0],
                          y=params[4][1],
                          is_categorical=categorical_list,
                          max_depth=max_depth,
                          columns=cols,
                          depth=self.depth + 1,
                          nb_features=self.nb_features,
                          weights=params[6])
        self.clean_attributes()
   
    def gini_split(self, idx_g_1, idx_g_2, cutoff, feature_idx, best_params):
        """Computes the gini index of the future split
        
        Args:
            idx_g_1: A torch tensor for the indices of the first group.
            idx_g_2: A torch tensor for the indices of the second group.
            cutoff: A float for the split value.
            feature_idx: A positive integer for the index of the feature to test.
            best_params: See function find_best_split
        """
        g_1 = self.y[idx_g_1].squeeze(1)
        g_2 = self.y[idx_g_2].squeeze(1)
        if self.weights is  None:
            #Gini index
            gini_g1 = (float(g_1.shape[0]) / self.size) * self.gini(g_1)
            gini_g2 = (float(g_2.shape[0]) / self.size) * self.gini(g_2)
        else:
            # Weighted gini index
            g_1_w = self.weights[idx_g_1]
            g_2_w = self.weights[idx_g_2]
            w_sum = self.weights.sum()
            gini_g1 = (g_1_w.sum() / w_sum) * self.gini(g_1, g_1_w)
            gini_g2 = (g_2_w.sum() / w_sum) * self.gini(g_2, g_2_w) 
            
        gini_split = (gini_g1 + gini_g2)
        if gini_split < best_params[0]:
            best_params[0] = gini_split.item()
            best_params[1] = feature_idx
            best_params[2] = cutoff.item()
            # If training a base learner of a random forest
            # pass the full data to child nodes
            if self.nb_features != 0:
                best_params[3] = [self.X_full[idx_g_1].squeeze(1), g_1]
                best_params[4] = [self.X_full[idx_g_2].squeeze(1), g_2]
            else:
                best_params[3] = [self.X[idx_g_1].squeeze(1), g_1]
                best_params[4] = [self.X[idx_g_2].squeeze(1), g_2]
            if self.weights is not None:
                # Gather weights of each groups to pass them to the child nodes
                # for their own weighted gini index
                best_params[5] = g_1_w
                best_params[6] = g_2_w
        return best_params
    
    def find_best_split(self):
        """Finds the best split
        
            Creates a parameter list to store the parameters of the best split
            It contains:
            0: best gini index
            1: column index of the best split
            2: value of the best split
            3: left group [X, y], less than equal to #3 or belongs to the class #3 if categorical
            4: right group [X, y], greater than #3 or does not belong to the class #3
            5 : weights of the first group
            6 : weights of the second group
        """
        best_params = [2, -1, -1, None, None, None, None]
        for i in range(self.X.shape[1]):
            vals = self.X[:, i]
            if self.is_categorical_list[i]:
                for cutoff in vals.unique():
                    idx_uv = (vals == cutoff).nonzero()
                    idx_uv_not = (vals != cutoff).nonzero()
                    best_params = self.gini_split(idx_uv, idx_uv_not, cutoff, i, best_params)
            else:
                for cutoff in vals.unique():
                    idx_leq = (vals <= cutoff).nonzero()
                    idx_ge = (vals > cutoff).nonzero()
                    best_params = self.gini_split(idx_leq, idx_ge, cutoff, i, best_params)
        return best_params
    
    def get_dict(self):
        """Returns a dictionary containing nodes and their information"""
        node_dict = {}
        if self.left is None and self.right is None:
            node_dict['pred'] = self.cutoff
        else:
            node_dict['cutoff'] = self.cutoff
            node_dict['feature'] = self.columns[self.col]
            node_dict['categorical'] = self.var_categorical
            node_dict['left'] = self.left.get_dict()
            node_dict['right'] = self.right.get_dict()
        return node_dict
   
    def predict(self, sample):
        """Takes a single input and predicts its class
        
            Follows the tree based on the conditions
        """
        if self.nb_features != 0:
            sample_in = sample[self.col_perm]
        else:
            sample_in = sample
        if self.left is None and self.right is None:
            return self.cutoff
        if self.var_categorical:
            if sample_in[self.col] == self.cutoff:
                return self.left.predict(sample)
            else:
                return self.right.predict(sample)
        else:
            if sample_in[self.col] <= self.cutoff:
                return self.left.predict(sample)
            else:
                return self.right.predict(sample)
            

class DecisionTreeClassifier(BaseEstimator):
    """Class for the Decision Tree Classifier
    
    Class that implements the sklearn methods
    Just a wrapper for our node class
    
    Attributes:
        max_depth: A positive integer for the maximum depth of the tree.
        columns: A list of the name of each column.
        nb_features: A positive integer for the number of features
        is_categorical: A list of boolean, 1 if the features is categorical,
    """
    def __init__(self, max_depth, is_categorical, columns, nb_features=0):
        """Inits the Decision Tree class
        
        Args:
            See attributes section.
        """
        if nb_features < 0:
            raise ValueError('negative integer passed to nb_features.')
        self.max_depth = max_depth
        # Wether or not each column is a categorical value
        self.is_categorical = is_categorical
        self.columns = columns
        self.root = None
        # Number of random features to select
        # Only used when building a random forest base learner
        # If 0 then train a decision tree using all the features availables
        self.nb_features = nb_features
    
    def fit(self, X, y, **kwargs):
        """Trains the model
        
        Needs to get the 'sample_weight' key in kwargs.
        Mandatory for using sklearn cross validation.
        
        Args:
            X: A torch tensor for the data.
            y: A torch tensor for the labels.
        """
        if self.nb_features > X.shape[1]:
            raise ValueError('parameter np_features should be less than equal to the number of features')
        if 'sample_weight' in kwargs.keys():
            weights = kwargs['sample_weight']
        else:
            weights = None
        self.root = Node(X, y, self.is_categorical, self.max_depth, self.columns, 0, self.nb_features, weights)
        
    def predict(self, X):
        """Predicts the labels of a batch of input samples"""
        if len(X.shape) == 0:
            return 'error: can not predict on empty input.'
        if len(X.shape) == 1:
            return self.root.predict(X)
        
        pred = torch.zeros((X.shape[0],), dtype=torch.int32)
        if self.root == None:
            return 'error: use the fit method before using predict.'
        for i in range(X.shape[0]):
            sample = X[i, :]
            pred[i] = self.root.predict(sample)
        return pred
    
    def __str__(self):
        """Method for printing the tree"""
        if self.root == None:
            return 'error: use the fit method to print the result of the training.'
        tree_dict = self.root.get_dict()
        pprint(tree_dict)
        return ''
    
dt = DecisionTreeClassifier(max_depth=3, is_categorical=False, columns=data.columns, nb_features=201123)
dt.fit(xtrain, ytrain)
print(dt)