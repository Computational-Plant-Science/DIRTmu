from itertools import izip

class Candidate_Graph:
    """
    Creates a graph from a candidate path.
    Can merge another candidate into this candidate 
    or remove another candidate from this candidate.
    """
    def __init__(self, path):
        self.graph = {}
        for first, second in izip(path, path[1:]):

            if first in self.graph:
                self.graph[first].append(second)
            else:
                self.graph[first] = [second]

            if second in self.graph:
                self.graph[second].append(first)
            else:
                self.graph[second] = [first]

    def merge(self, other):
        """
        Merges other graph into this graph.
        Can have multiple edges between two vertices.
        """
        for vertex, neighbbor_list in other.graph.items():
            for neighbor in neighbbor_list:

                if vertex in self.graph:
                    self.graph[vertex].append(neighbor)
                else:
                    self.graph[vertex] = [neighbor]

    def remove(self, other):
        """
        Removes edges in other graph from this graph.
        """
        for vertex, neighbbor_list in other.graph.items():
            if vertex in self.graph:
                for neighbor in neighbbor_list:
                    try:
                        self.graph[vertex].remove(neighbor)
                    except:
                        pass # Do not remove vertex
                if len(self.graph[vertex]) == 0:
                    self.graph.pop(vertex, None)

    def get_path(self):
        v_start = self.start_vertex()
        path = list(self.bfs(v_start))
        return path

    def bfs(self, s):
        """
        Breadth first search starting from vertex s
        """

        visited = {v: False for v in self.graph.keys()}
        queue = []

        queue.append(s)
        visited[s] = True

        while queue:
 
            s = queue.pop(0)
            yield s
 
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True


    def start_vertex(self):
        for v,l in self.graph.items():
            if len(l) == 1:
                return v
        return None

    def all_degree_one(self):
        vertices = []
        for v,l in self.graph.items():
            if len(l) == 1:
                vertices.append(v)
        return vertices

    def copy(self):
        c = Candidate_Graph([])
        c.graph = {key: list(value) for key, value in self.graph.items()}
        return c


if __name__ == '__main__':

    path_1 = [1,2,3,4,5]
    path_2 = [3,4,5,6,7,8,9]
    path_3 = [7,8,9,10,11]

    g1 = Candidate_Graph(path_1)
    print g1.get_path()

    g2 = Candidate_Graph(path_2)
    print g2.get_path()

    g3 = Candidate_Graph(path_3)
    print g3.get_path()

    g1.merge(g2)
    g1.merge(g3)
    print g1.graph
    print g1.get_path()
    print g1.all_degree_one()





