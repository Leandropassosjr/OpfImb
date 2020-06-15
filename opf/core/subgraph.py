import numpy as np

import opf.stream.loader as loader
import opf.stream.parser as p
import opf.utils.constants as c
import opf.utils.exception as e
import opf.utils.logging as l
from opf.core.node import Node

logger = l.get_logger(__name__)


class Subgraph:
    """A Subgraph class is used as a collection of Nodes and the basic structure to work with OPF.

    """

    def __init__(self, X=None, Y=None, idx_sample=None, from_file=None):
        """Initialization method.

        Args:
            X (np.array): Array of features.
            Y (np.array): Array of labels.
            from_file (bool): Whether Subgraph should be directly created from a file.

        """

        # Number of nodes
        self.n_nodes = 0

        # Number of features
        self.n_features = 0

        # List of nodes
        self.nodes = []

        # List of indexes of ordered nodes
        self.idx_nodes = []

        # Whether the subgraph is trained or not
        self.trained = False

        # Checks if data should be loaded from a file
        if from_file:
            # Loads the data
            X, Y = self._load(from_file)

        # Checks if data has been properly loaded
        if X is not None:
            # Checks if labels are provided or not
            if Y is None:
                # If not, creates an empty numpy array
                Y = np.ones(len(X), dtype=int)
            if idx_sample is None:
                # If not, creates an empty numpy array
                idx_sample = np.zeros(len(X), dtype=int)

            # Now, we need to build this class up
            self._build(X, Y, idx_sample)

        # If data could not be loaded
        else:
            logger.error('Subgraph has not been properly created.')

    @property
    def n_nodes(self):
        """int: Number of nodes.

        """

        return len(self.nodes)

    @n_nodes.setter
    def n_nodes(self, n_nodes):
        if not isinstance(n_nodes, int):
            raise e.TypeError('`n_nodes` should be an integer')
        if n_nodes < 0:
            raise e.ValueError('`n_nodes` should be >= 0')

        self._n_nodes = n_nodes

    @property
    def n_features(self):
        """int: Number of features.

        """

        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        if not isinstance(n_features, int):
            raise e.TypeError('`n_features` should be an integer')
        if n_features < 0:
            raise e.ValueError('`n_features` should be >= 0')

        self._n_features = n_features

    @property
    def nodes(self):
        """list: List of Nodes that belongs to the Subgraph.

        """

        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        if not isinstance(nodes, list):
            raise e.TypeError('`nodes` should be a list')

        self._nodes = nodes

    @property
    def idx_nodes(self):
        """list: List of indexes of ordered nodes.

        """

        return self._idx_nodes

    @idx_nodes.setter
    def idx_nodes(self, idx_nodes):
        if not isinstance(idx_nodes, list):
            raise e.TypeError('`idx_nodes` should be a list')

        self._idx_nodes = idx_nodes

    @property
    def trained(self):
        """bool: Indicate whether the subgraph is trained.

        """

        return self._trained

    @trained.setter
    def trained(self, trained):
        if not isinstance(trained, bool):
            raise e.TypeError('`trained` should be a boolean')

        self._trained = trained

    def _load(self, file_path):
        """Loads and parses a dataframe from a file.

        Args:
            file_path (str): File to be loaded.

        Returns:
            Arrays holding the features and labels.

        """

        # Getting file extension
        extension = file_path.split('.')[-1]

        # Check if extension is .csv
        if extension == 'csv':
            # If yes, call the method that actually loads csv
            data = loader.load_csv(file_path)

        # Check if extension is .txt
        elif extension == 'txt':
            # If yes, call the method that actually loads txt
            data = loader.load_txt(file_path)

        # Check if extension is .json
        elif extension == 'json':
            # If yes, call the method that actually loads json
            data = loader.load_json(file_path)

        # If extension is not recognized
        else:
            # Raises an ArgumentError exception
            raise e.ArgumentError(
                'File extension not recognized. It should be `.csv`, `.json` or `.txt`')

        # Parsing array
        X, Y = p.parse_loader(data)

        return X, Y

    def _build(self, X, Y, idx_sample):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            X (np.array): Features array.
            Y (np.array): Labels array.

        """

        # Iterate over every possible sample
        for i, (feature, label, index_sample) in enumerate(zip(X, Y, idx_sample)):
            # Creates a Node structure
            node = Node(i, label.item(),index_sample.item(), feature)

            # Appends the node to the list
            self.nodes.append(node)

        # Calculates the number of features
        self.n_features = self.nodes[0].features.shape[0]

    def destroy_arcs(self):
        """Destroy the arcs present in the subgraph.

        """

        # For every possible node
        for i in range(self.n_nodes):
            # Reset the number of adjacent nodes
            self.nodes[i].n_plateaus = 0

            # Resets the list of adjacent nodes
            self.nodes[i].adjacency = []

    def mark_nodes(self, i):
        """Marks a node and its whole path as relevant.

        Args:
            i (int): An identifier of the node to start the marking.

        """

        # While the node still has a predecessor
        while self.nodes[i].pred != c.NIL:
            # Marks current node as relevant
            self.nodes[i].relevant = c.RELEVANT

            # Gathers the predecessor node of current node
            i = self.nodes[i].pred

        # Marks the first node as relevant
        self.nodes[i].relevant = c.RELEVANT

    def reset(self):
        """Resets the subgraph predecessors and arcs.

        """

        # For every possible node
        for i in range(self.n_nodes):
            # Resets its predecessor
            self.nodes[i].pred = c.NIL

            # Resets whether its relevant or not
            self.nodes[i].relevant = c.IRRELEVANT

        # Destroys the arcs
        self.destroy_arcs()
