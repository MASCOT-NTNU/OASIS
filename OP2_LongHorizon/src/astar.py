



class node():
    """A box class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.x, self.y = position

        self.g = 0
        self.h = 0
        self.f = 0

    # def __eq__(self, other):
    #     return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given board"""

    # Create start and end node
    start_node = node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if (current_node.x == end_node.x) and (current_node.y == end_node.y):
            path = []
            current = current_node
            while current is not None:
                path.append([current.x, current.y])
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        for new_position in [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]]:  # Adjacent squares

            # Get node position
            node_position = (current_node.x + new_position[0], current_node.y + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if (child.x == closed_child.x) and (child.y == closed_child.y):
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.x - end_node.x) ** 2) + ((child.y - end_node.y) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if (child.x == open_node.x) and (child.y == open_node.y) and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def main():

    # board = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    board = [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]]
    start = (0, 0)
    end = (6, 6)

    path = astar(board, start, end)
    print(path)


if __name__ == '__main__':
    main()









