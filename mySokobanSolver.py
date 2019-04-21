'''
    2019 CAB320 Sokoban assignment
The functions and classes defined in this module will be called by a marker script.
You should complete the functions and classes according to their specified interfaces.
You are not allowed to change the defined interfaces.
That is, changing the formal parameters of a function will break the
interface and triggers to a fail for the test of your code.
# by default does not allow push of boxes on taboo cells
SokobanPuzzle.allow_taboo_push = False
# use elementary actions if self.macro == False
SokobanPuzzle.macro = False
'''

# you have to use the 'search.py' file provided
# as your code will be tested with this specific file
import search
import sokoban

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    return [(9776460, 'Rune', 'Leistad'), (10405127, 'Jenny', 'Bogen Griffiths')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Takes an array of coordinate and finds corners that has their 90 degree angle wall
inside the warehouse. Yields an array in the format [('DIRECTION-CORNER', [x, y])]
@param walls A tuple of coordinates for each of the walls in the warehouse
'''
def find_corners(walls):
    for wall in walls:
        corner = False
        if(tuple([wall[0] + 1, wall[1]]) in walls and tuple([wall[0], wall[1] + 1]) in walls):
            yield 'SE-CORNER', [wall[0],wall[1]] # South east
            corner = True
        if(tuple([wall[0] - 1, wall[1]]) in walls and tuple([wall[0], wall[1] + 1]) in walls):
            yield 'SW-CORNER', [wall[0],wall[1]] # South west corner
            corner = True
        if(tuple([wall[0] - 1, wall[1]]) in walls and tuple([wall[0], wall[1] - 1]) in walls):
            yield 'NW-CORNER', [wall[0],wall[1]] # North west corner
            corner = True
        if(tuple([wall[0] + 1, wall[1]]) in walls and tuple([wall[0], wall[1] - 1]) in walls):
            yield 'NE-CORNER', [wall[0],wall[1]] # North east corner
            corner = True
        if not corner:
            yield 'WALL', [wall[0],wall[1]]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Identifying whether or not a cell is indside or outside of the walls in the
warehouse
@param warehouse: A Warehouse object
@param cell: The coordiantes of the cell you want to check
@param X: The max(x) coordinate
@param Y: The max(Y) coordinate
@return
    True if the given cell has a wall in the north, east, south and west
    False if no wall was found in any one of the directions
'''
def cell_inside(warehouse, cell, X, Y):
    wall_count = 0
    directions = [[1, 0], [0, 1], [0, -1], [-1, 0]] # [east, south, north, west]

    for dir in directions:
        x = cell[0]
        y = cell[1]
        # While within the bounds of the warehouse
        while(x > -1 and x < X and y > -1 and y < Y):
            x += dir[0]
            y += dir[1]
            if tuple([x, y]) in warehouse.walls:
                wall_count += 1
                break


    return wall_count == 4

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
Takes a list of tuples and creates a shallow copy.
@param copy_list Array to be copied / cloned
@return Returns a shallow copy of copy_list
'''
def shallow_copy_tuples(copy_list):
    clone = []
    for elem in copy_list:
        clone.append(tuple([elem[0], elem[1]]))

    return clone

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def taboo_cells(warehouse):
    '''
    Identify the taboo cells of a warehouse. A cell inside a warehouse is
    called 'taboo' if whenever a box get pushed on such a cell then the puzzle
    becomes unsolvable.
    When determining the taboo cells, you must ignore all the existing boxes,
    simply consider the walls and the target cells.
    Use only the following two rules to determine the taboo cells;
     Rule 1: if a cell is a corner inside the warehouse and not a target,
             then it is a taboo cell.
     Rule 2: all the cells between two corners inside the warehouse along a
             wall are taboo if none of these cells is a target.
    @param warehouse: a Warehouse object
    @return
       A string representing the puzzle with only the wall cells marked with
       an '#' and the taboo cells marked with an 'X'.
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.
    '''

    wt_cells = warehouse.__str__()
    legal_characters = ['#', ' ', '\n']

    # Replacing everything that isn't a wall, blank space or new line with ' '
    for cell in wt_cells:
        if cell not in legal_characters:
            wt_cells = wt_cells.replace(cell, ' ')

    # Finding the maxiumum dimentions of the warehouse
    X,Y = zip(*warehouse.walls)
    x_size, y_size = max(X), max(Y)
    # Converting string to a list to easier replace cell content
    taboo = list(wt_cells)
    line_size = x_size + 2 # The size of each line in the string (inkl. '\n')
    walls_and_corners = find_corners(warehouse.walls.copy())

    # Looping through all the walls, and for those who are corners set taboo cell
    # iff cell is not a target
    for wall in walls_and_corners:
        x = wall[1][0]
        y = wall[1][1]
        if wall[0] == 'SE-CORNER':
            # Checking that the corner is not a target and is inside the warehouse
            if tuple([x+1, y+1]) not in warehouse.targets and \
            cell_inside(warehouse, tuple([x+1, y+1]), x_size, y_size):
                taboo[(x+1) + ((y+1) * line_size)] = 'X'
        elif wall[0] == 'SW-CORNER':
            # Checking that the corner is not a target and is inside the warehouse
            if tuple([x-1, y+1]) not in warehouse.targets and \
            cell_inside(warehouse, tuple([x-1, y+1]), x_size, y_size):
                taboo[(x-1) + ((y+1) * line_size)] = 'X'
        elif wall[0] == 'NE-CORNER':
            # Checking that the corner is not a target and is inside the warehouse
            if tuple([x+1, y-1]) not in warehouse.targets and \
            cell_inside(warehouse, tuple([x+1, y-1]), x_size, y_size):
                taboo[(x+1) + ((y-1) * line_size)] = 'X'
                # Checking that the corner is not a target and is inside the warehouse
        elif wall[0] == 'NW-CORNER':
            if tuple([x-1, y-1]) not in warehouse.targets and \
            cell_inside(warehouse, tuple([x-1, y-1]), x_size, y_size):
                taboo[(x-1) + ((y-1) * line_size)] = 'X'

    # Finding entire walls to be marked taboo
    taboo_walls = find_corners(warehouse.walls.copy())

    # Find taboo corner cells. When found, find all neighbouring walls.
    # for each wall go in x direction untill corner or blank space is found, or
    # target is found next to the wall.
    # If matching corner is found and wall has no target -> set entire wall taboo.
    for wall in taboo_walls:
        directions = []
        # while finding taboo walls this loops through both x and y direction.
        # Therefore there is no need to check 'SW-CORNER' and 'NE-CORNER'
        if wall[0] == 'SE-CORNER':
            directions = [[1, 0], [0, 1]] # [0] = east, [1] = south
        if wall[0] == 'NW-CORNER':
            directions = [[-1, 0], [0, -1]] # [0] = west, [1] = north

        for dir in directions:
            next_wall = tuple([wall[1][0] + dir[0], wall[1][1] + dir[1]])
            tb_cell = tuple([wall[1][0] + sum(dir), wall[1][1] + sum(dir)])
            # As long as it's still a possible taboo wall
            while (next_wall in warehouse.walls and
            (tb_cell not in warehouse.targets and tb_cell not in warehouse.walls)):
                next_wall = tuple([next_wall[0] + dir[0], next_wall[1] + dir[1]])
                tb_cell = tuple([tb_cell[0] + dir[0], tb_cell[1] + dir[1]])
            # If while loop breaks out and these condictions match, then its a taboo wall
            if next_wall in warehouse.walls and tb_cell in warehouse.walls:
                tb_cell = tuple([wall[1][0] + sum(dir), wall[1][1] + sum(dir)])
                while tb_cell not in warehouse.walls:
                    taboo[(tb_cell[0]) + ((tb_cell[1]) * line_size)] = 'X'
                    tb_cell = tuple([tb_cell[0] + dir[0], tb_cell[1] + dir[1]])


    return "".join(taboo)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Finds the manhattan distance between two points.
@param a First point
@param b Second point
@return Returns the sum of (a0 - b0) + (a1 - b1)
'''
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

'''
Finds the euclidian distance between two points
@param a First point
@param b Second point
@return Returns euclidian distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)
'''
def euclidian_distance(a, b):
    return pow(pow(abs(a[0]-b[0]), 2) + pow(abs(a[1]-b[1]), 2), 0.5)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.
    Your implementation should be fully compatible with the search functions of
    the provided module 'search.py'.
    Each instance should have at least the following attributes
    - self.allow_taboo_push
    - self.macro
    When self.allow_taboo_push is set to True, the 'actions' function should
    return all possible legal moves including those that move a box on a taboo
    cell. If self.allow_taboo_push is set to False, those moves should not be
    included in the returned list of actions.
    If self.macro is set True, the 'actions' function should return
    macro actions. If self.macro is set False, the 'actions' function should
    return elementary actions.
    '''

    def __init__(self, warehouse, allow_taboo = False, macro = False):
        self.allow_taboo_push = True if allow_taboo else False
        self.macro = True if macro else False
        self.taboo = taboo_cells(warehouse)
        self.initial = warehouse

        # Without the shallow copy the warehouse.boxes and self.goal.boxes would
        # point to the same memory location.
        self.goal = warehouse.copy(warehouse.worker, shallow_copy_tuples(warehouse.boxes))
        counter = 0

        for target in self.goal.targets:
            for box in self.goal.boxes:
                if box not in self.goal.targets:
                    self.goal.boxes.remove(box)
                    self.goal.boxes.append(self.goal.targets[counter])
                    counter += 1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """


        action_list = []
        right = tuple([state.worker[0] + 1, state.worker[1]])
        left = tuple([state.worker[0] - 1, state.worker[1]])
        down = tuple([state.worker[0], state.worker[1] + 1])
        up = tuple([state.worker[0], state.worker[1] - 1])
        # Creating a 2d array of the taboo cells
        taboo_2d = []
        taboo_2d.append([])
        i = 0
        for line in self.taboo:
            for char in line:
                if char == '\n':
                    taboo_2d[i].append(char)
                    taboo_2d.append([])
                    i += 1
                else:
                    taboo_2d[i].append(char)

        # This section could've been shortened by about 3/4ths for better
        # readability. We didn't have time to implement the change.
        if not self.macro: # Elementary actions
            if right not in state.walls:
                if right in state.boxes:
                    next_right = tuple([right[0] + 1, right[1]])
                    # If box is to the right of the worker, check if we can move it
                    if (next_right not in state.boxes and next_right not in state.walls):
                        if not self.allow_taboo_push:
                             if taboo_2d[next_right[1]][next_right[0]] != 'X':
                                 action_list.append('Right')
                        else:
                            action_list.append('Right')
                else:
                    action_list.append('Right')

            if left not in state.walls:
                if left in state.boxes:
                    next_left = tuple([left[0] - 1, left[1]])
                    if (next_left not in state.boxes and next_left not in state.walls):
                        if not self.allow_taboo_push:
                             if taboo_2d[next_left[1]][next_left[0]] != 'X':
                                 action_list.append('Left')
                        else:
                            action_list.append('Left')
                else:
                    action_list.append('Left')

            if down not in state.walls:
                if down in state.boxes:
                    next_down = tuple([down[0], down[1] + 1])
                    if (next_down not in state.boxes and next_down not in state.walls):
                        if not self.allow_taboo_push:
                             if taboo_2d[next_down[1]][next_down[0]] != 'X':
                                 action_list.append('Down')
                        else:
                            action_list.append('Down')
                else:
                    action_list.append('Down')

            if up not in state.walls:
                if up in state.boxes:
                    next_up = tuple([up[0], up[1] - 1])
                    if (next_up not in state.boxes and next_up not in state.walls):
                        if not self.allow_taboo_push:
                             if taboo_2d[next_up[1]][next_up[0]] != 'X':
                                 action_list.append('Up')
                        else:
                            action_list.append('Up')
                else:
                    action_list.append('Up')


        # If macro moves, check which boxes can be moved in what direction
        else:
            macro_moves = state.boxes

            for cell in macro_moves:
                right = tuple([cell[0] + 1, cell[1]])
                left = tuple([cell[0] - 1, cell[1]])
                down = tuple([cell[0], cell[1] + 1])
                up = tuple([cell[0], cell[1] - 1])

                # Checking left and right at the same time because if worker can stand
                # on the left side but there is a wall on the right, then you cant
                # push the box in that dirrection. Same for up and down.
                if (right not in state.walls and right not in state.boxes and
                left not in state.walls and left not in state.boxes):

                    if can_go_there(state, right):
                        if self.allow_taboo_push:
                            action_list.append([right, 'Right'])
                        elif taboo_2d[left[1]][left[0]] != 'X':
                            action_list.append([right, 'Right'])
                    if can_go_there(state, left):
                        if self.allow_taboo_push:
                            action_list.append([left, 'Left'])
                        elif taboo_2d[right[1]][right[0]] != 'X':
                            action_list.append( [left, 'Left'])
                        action_list.append([left, 'Left'])


                if (down not in state.walls and down not in state.boxes and
                up not in state.walls and up not in state.boxes):

                    if can_go_there(state, down):
                        if self.allow_taboo_push:
                            action_list.append([down, 'Down'])
                        elif taboo_2d[up[1]][up[0]] != 'X':
                            action_list.append([down, 'Down'])
                    if can_go_there(state, up):
                        if self.allow_taboo_push:
                            action_list.append([up, 'Up'])
                        elif taboo_2d[down[1]][down[0]] != 'X':
                            action_list.append( [up, 'Up'])


        return action_list


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    '''
    Needed by the search algorithms to expand the tree / graph. This function
    performs an action on a state and returns the result of the action.
    @param state A warehouse object representing the current state of the problem
    @param action The action to be performed on the state.
    @return Returns the state which occurs after action has been performed on the state
    '''
    def result(self, state, action):
        next_state = state.copy(tuple([state.worker[0], state.worker[1]]), shallow_copy_tuples(state.boxes))

        assert action in self.actions(state)

        if self.macro: #action = ((r1,c1), a1)
            micro_actions = find_macro_moves(next_state, action[0])
            worker_pos = tuple([next_state.worker[0], next_state.worker[1]])
            # Moves the worker to the cell that was found in actions
            for move in micro_actions:
                if move == 'Right':
                    worker_pos = tuple([worker_pos[0] + 1, worker_pos[1]])
                elif move == 'Left':
                    worker_pos = tuple([worker_pos[0] - 1, worker_pos[1]])
                elif move == 'Down':
                    worker_pos = tuple([worker_pos[0], worker_pos[1] + 1])
                elif move == 'Up':
                    worker_pos = tuple([worker_pos[0], worker_pos[1] - 1])

            next_state.worker = worker_pos
            # right, left, down and up indicates on what side of the box the
            # worker stands on before pushing it
            right = tuple([worker_pos[0]-1, worker_pos[1]])
            left = tuple([worker_pos[0]+1, worker_pos[1]])
            down = tuple([worker_pos[0], worker_pos[1]-1])
            up = tuple([worker_pos[0], worker_pos[1]+1])

            if action[1] == 'Right':
                next_state.boxes.append(tuple([right[0]-1, right[1]]))
                next_state.boxes.remove(right)
                next_state.worker = right
            elif action[1] == 'Left':
                next_state.boxes.append(tuple([left[0]+1, left[1]]))
                next_state.boxes.remove(left)
                next_state.worker = left
            elif action[1] == 'Down':
                next_state.boxes.append(tuple([down[0], down[1]-1]))
                next_state.boxes.remove(down)
                next_state.worker = down
            elif action[1] == 'Up':
                next_state.boxes.append(tuple([up[0], up[1]+1]))
                next_state.boxes.remove(up)
                next_state.worker = up
            return next_state

        else: #elementary action
            x = 0
            y = 0

            if action == 'Up':
                y -= 1
            elif action == 'Down':
                y += 1
            elif action == 'Left':
                x -= 1
            elif action == 'Right':
                x += 1

            pos_worker = tuple([state.worker[0] + x, state.worker[1] + y])
            moved = False
            # Deals with when the worker is pushing a box. actions() already
            # checks that the move is valid.
            for box in next_state.boxes:
                # Finds the correct box
                if box == pos_worker:
                    next_state.boxes.remove(box)
                    next_state.boxes.append(tuple([box[0] + x, box[1] + y]))
                    moved = True
                    break
            if moved:
                next_state.worker = pos_worker
            else:
                next_state.worker = tuple([next_state.worker[0] + x,next_state.worker[1] + y])
            return next_state

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    '''
    Checks if the given state is equal to the goal state.
    @param state State of the warehouse to check
    @return True if warehouse is in a goal state, False if not.
    '''
    def goal_test(self, state):
        # Goal is reached if every box is placed on a goal box location
        for box in state.boxes:
            if box not in self.goal.boxes:
                # Any box not on a target means that state is not reached its goal
                return False
        return True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    '''
    Heuristics for the current state. It is measured by finding how far away each
    box is away from its nearest target and how close worker is to a box.
    h = sum(shortest distance of each box to target +
    distance between worker and closest box)
    @param n Node that contains the state in which we want to find h
    @return The sum of shortest distance to target for each box + worker to nearest box
    '''
    def h(self, n):
        dist = 0
        closest = 100
        for i, box in enumerate(n.state.boxes):
            dist_min = euclidian_distance(n.state.boxes[i], n.state.targets[0])
            # Finding distance between each box and closest target
            for target in n.state.targets:
                temp_euc_dist = euclidian_distance(n.state.boxes[i], target)
                if temp_euc_dist < dist_min:
                    dist_min = temp_euc_dist
            dist += dist_min
        # Finding which of the boxes worker i closest to
        for box in n.state.boxes:
            ed = euclidian_distance(box, n.state.worker)
            if ed < closest:
                closest = ed
        return dist+closest

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def check_action_seq(warehouse, action_seq):
    '''
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
    @param warehouse: a valid Warehouse object
    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
    @return
        The string 'Failure', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall.
        Otherwise, if all actions were successful, return
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''

    # Recursive function. First checking if it reached the end goal
    if len(action_seq) < 1:
        return warehouse.__str__()

    puzzle = SokobanPuzzle(warehouse)
    # If action sequence still has values, pop first in list
    action = action_seq.pop(0)
    if action not in puzzle.actions(warehouse):
        return 'Failure'
    # Executing the action on the warehouse
    result = puzzle.result(warehouse, action)
    return check_action_seq(result, action_seq)



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
Helper class to find if a worker can get from a to b without moving any boxes.
'''
class SokobanMacroPuzzle(search.Problem):
    '''
    Constructor of SokobanMacroPuzzle takes a warehouse and a goal to help
    define a partial answer to the SokobanPuzzle
    '''
    def __init__(self, warehouse, goal):
        self.initial = warehouse
        self.goal = goal
        warehouse.copy(warehouse.worker, shallow_copy_tuples(warehouse.boxes))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Actions finds any possible action given a state. These actions can not include
    moving boxes.
    @param state The puzzle's current state.
    @return Returns a list of possible actions in the given state
    '''
    def actions(self, state):
        action_list = []
        right = tuple([state.worker[0] + 1, state.worker[1]])
        left = tuple([state.worker[0] - 1, state.worker[1]])
        down = tuple([state.worker[0], state.worker[1] + 1])
        up = tuple([state.worker[0], state.worker[1] - 1])

        if right not in state.boxes and right not in state.walls:
            action_list.append('Right')
        if left not in state.boxes and left not in state.walls:
            action_list.append('Left')
        if down not in state.boxes and down not in state.walls:
            action_list.append('Down')
        if up not in state.boxes and up not in state.walls:
            action_list.append('Up')

        return action_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Takes an action and applies it to the given state.
    @param state Current state
    @return The state after action has been performed
    '''
    def result(self, state, action):
        x = 0
        y = 0

        if action == 'Up':
            y -= 1
        elif action == 'Down':
            y += 1
        elif action == 'Left':
            x -= 1
        elif action == 'Right':
            x += 1

        pos_worker = tuple([state.worker[0] + x, state.worker[1] + y])
        return state.copy(pos_worker, shallow_copy_tuples(state.boxes)) # next_state

    '''
    Finds out if state equals goal state which is when worker has reached the goal
    @param state Warehouse object that represents the state of a warehouse
    @return Returns True if worker has reached its goal. False if not
    '''
    def goal_test(self, state):
        return self.goal.worker == state.worker

    '''
    Heuristics for the current state. It is measured by finding how far away the
    worker is from reaching its goal state.
    @param n Node that contains the state in which we want to find h
    @return The sum of shortest distance between worker and goal. Walls are ignored
    '''
    def h(self, n):
        return euclidian_distance(self.goal.worker, n.state.worker)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_elem(warehouse):
    '''
    This function should solve using elementary actions
    the puzzle defined in a file.
    @param warehouse: a valid Warehouse object
    @return
        If puzzle cannot be solved return the string 'Impossible'
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''

    node = search.astar_graph_search(SokobanPuzzle(warehouse))
    if node == None:
        return 'Impossible'
    action_list = node.solution()
    return action_list


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
This function find the path for a worker from warehouse.worker to dst
@param warehouse State of the warehouse
@param dst Tuple containing the x and y position of the worker's goal
@return Returns the shortest way for a worker to get from a to b. If not found it
returns an empty list
'''
def find_macro_moves(warehouse, dst):
    # Searches through a warehouse with dst as worker's dst as a goal state
    actions = search.astar_graph_search(SokobanMacroPuzzle(warehouse, warehouse.copy(dst)))
    solution = [] if actions is None else actions.solution()
    return solution

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def can_go_there(warehouse, dst):
    '''
    Determine whether the worker can walk to the cell dst=(row,column)
    without pushing any box.
    @param warehouse: a valid Warehouse object
    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    '''

    if warehouse.worker == dst:
        return True
    return len(find_macro_moves(warehouse, dst)) > 0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_macro(warehouse):
    '''
    Solve using macro actions the puzzle defined in the warehouse passed as
    a parameter. A sequence of macro actions should be
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ]
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes to the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.
    @param warehouse: a valid Warehouse object
    @return
        If puzzle cannot be solved return the string 'Impossible'
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    '''

    macro_sokoban = SokobanPuzzle(warehouse, macro = True)
    node = search.astar_graph_search(macro_sokoban)
    return node.solution()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
