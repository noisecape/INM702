from Game import Agent, Node, GameManager
import unittest


class TestAgent(unittest.TestCase):



    def test_a_star(self):
        graph = [Node.Node(0, (0, 0)), Node.Node(0, (0, 1)), Node.Node(1, (0, 2)), Node.Node(7, (0, 3)),
                 Node.Node(1, (1, 0)), Node.Node(3, (1, 1)), Node.Node(4, (1, 2)), Node.Node(5, (1, 3))]
        list_of_nodes = []
        for node in graph:
            list_of_nodes.append(node)

        self.__find_neighbors(list_of_nodes)
        agent = Agent.Agent()
        agent.strategy = 'A_STAR'
        path = agent.apply_a_star(graph)
        solution = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)]
        assert path == solution

    def test_dijkstra(self):
        graph = [Node.Node(0, (0, 0)), Node.Node(0, (0, 1)), Node.Node(1, (0, 2)), Node.Node(7, (0, 3)),
                 Node.Node(1, (1, 0)), Node.Node(3, (1, 1)), Node.Node(4, (1, 2)), Node.Node(5, (1, 3))]
        list_of_nodes = []
        for node in graph:
            list_of_nodes.append(node)

        self.__find_neighbors(list_of_nodes)
        agent = Agent.Agent()
        agent.strategy = 'DIJKSTRA'
        path = agent.apply_dijkstra(graph)
        solution = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)]
        assert path == solution

    def __find_neighbors(self, list_of_nodes):
        """
        This function iterate through the adjacency list and find all the neighbours for each node.
        :return:
        void
        """
        for node in list_of_nodes:
            x_pos = node.location[0]
            y_pos = node.location[1]
            if x_pos - 1 >= 0:
                # find the node in the list of nodes
                # add it as a neighbor of the current node
                neighbor = self.__find_neighbor_at(x_pos - 1, y_pos, list_of_nodes)
                node.add_neighbor(neighbor)
            if x_pos + 1 <= 1:
                neighbor = self.__find_neighbor_at(x_pos + 1, y_pos, list_of_nodes)
                node.add_neighbor(neighbor)
            if y_pos - 1 >= 0:
                neighbor = self.__find_neighbor_at(x_pos, y_pos - 1, list_of_nodes)
                node.add_neighbor(neighbor)
            if y_pos + 1 <= 3:
                neighbor = self.__find_neighbor_at(x_pos, y_pos + 1, list_of_nodes)
                node.add_neighbor(neighbor)

    def __find_neighbor_at(self, x, y, list_of_nodes):
        """
        Find the neighbor at a specific location.
        :param x: the x coordinate of the neighbor to be found
        :param y: the y coordinate of the neighbor to be found
        :return:
        node: the neighbor of a current node with the exact location (x,y)
        """
        for node in list_of_nodes:
            if node.location == (x, y):
                return node