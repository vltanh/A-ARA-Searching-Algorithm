import heapq
import math
from time import sleep
import time
import multiprocessing
import argparse
from tkinter import *
from tkinter import filedialog, simpledialog, messagebox
import os
import random

class PriorityQueue:
    def __init__(self):
        self.heap = []
    def isEmpty(self):
        return not self.heap
    def push(self, n):
        heapq.heappush(self.heap, n)
    def pop(self):
        if self.isEmpty():
            raise KeyError('Empty queue!')
        return heapq.heappop(self.heap)
    def peek(self):
        if self.isEmpty():
            raise KeyError('Empty queue!')
        return self.heap[0]
    def show(self):
        print(list(self.heap))

class SearchAlgorithmFactory:
    factories = {}
    
    @staticmethod
    def addFactory(id, searchAlgorithm):
        SearchAlgorithmFactory.factories.put[id] = searchAlgorithm
    
    @staticmethod
    def getAlgorithm(id):
        if not id in SearchAlgorithmFactory.factories:
            SearchAlgorithmFactory.factories[id] = eval('{}.Factory()'.format(id))
        return SearchAlgorithmFactory.factories[id].create()
    
class SearchAlgorithm:
    pass
    
class AStar(SearchAlgorithm):
    def __init__(self, board, heuristic, Epsilon):
        self.board = board
        self.heuristic = heuristic
        self.Epsilon = Epsilon

    def search(self):
        open_set = PriorityQueue()
        close_set = []
        cost = {}
        parent = {}
        
        open_set.push((0, self.board.start))
        cost[self.board.start] = 0
        parent[self.board.start] = None
        
        while not open_set.isEmpty():
            curCost, curCell = open_set.pop()
            
            if curCell == self.board.goal:
                break
                
            close_set.append(curCell)
            
            for nextCell in self.board.getNeighbors(curCell):
                if nextCell not in close_set:
                    g = cost[curCell] + 1
                    h = self.Epsilon * self.heuristic(nextCell, self.board.goal)
                    f = g + h
                    if nextCell not in cost or cost[nextCell] > g:
                        cost[nextCell] = g
                        open_set.push((f, nextCell))
                        parent[nextCell] = curCell
        
        return parent
        
    class Factory:
        def create(self): return AStar

class ARAStar(SearchAlgorithm):
    def __init__(self, board, heuristic, startingEpsilon):
        self.open_set = PriorityQueue()
        self.close_set = []
        self.incons_set = []
        
        self.cost = {}
        self.parent = {}

        self.board = board
        self.heuristic = heuristic
        self.Epsilon = startingEpsilon

    def key(self, s):
        return self.cost[s] + self.Epsilon * self.heuristic(s, self.board.goal)
        
    def ImprovePath(self):
        while not self.open_set.isEmpty() and self.key(self.board.goal) > self.open_set.peek()[0]:
            curCost, curCell = self.open_set.pop()
            self.close_set.append(curCell)
            
            for nextCell in self.board.getNeighbors(curCell):
                if nextCell not in self.cost or self.cost[nextCell] > self.cost[curCell] + 1:  
                    self.cost[nextCell] = self.cost[curCell] + 1
                    self.parent[nextCell] = curCell
                    if nextCell not in self.close_set:
                        f = self.key(nextCell)
                        self.open_set.push((f, nextCell))
                    else:
                        self.incons_set.append(nextCell)
    
    def searchUtil(self, records):
        self.cost[self.board.start] = 0
        self.parent[self.board.start] = None
        self.cost[self.board.goal] = 1e9
        self.open_set.push((self.key(self.board.start), self.board.start))
        
        while self.Epsilon >= 1:
            self.ImprovePath()
            path = Path(self.parent, self.board.start, self.board.goal)
            records.append((self.Epsilon, self.parent))
            self.Epsilon -= 0.03125
            
            common_set = set()
            while not self.open_set.isEmpty():
                common_set.add(self.open_set.pop()[1])
            for cell in self.incons_set:
                common_set.add(cell)
            for cell in common_set:
                self.open_set.push((self.key(cell), cell))
                
            self.incons_set = []
            self.close_set = []

    def search(self):
        manager = multiprocessing.Manager()
        records = manager.list()

        p = multiprocessing.Process(target = self.searchUtil, name = "ARA", args = (records,))
        p.start()
        p.join(0.118)

        if p.is_alive():
            p.terminate()
            p.join()

        parent = {}
        for record in records:
            E, parent = record
            path = Path(parent, self.board.start, self.board.goal)
            length = path.getLength()
            print('E = {0:.9f}, len = {1}'.format(E, length))

        return parent

    class Factory:
        def create(self): return ARAStar

class Path:
    def __init__(self, parent, start, goal):
        self.start = start
        self.goal = goal
        self.getPathFromParentSet(parent)
    
    def getPathFromParentSet(self, parent):
        self.path = []
        
        cur = self.goal
        if (cur not in parent): 
            return
        
        while (cur != self.start):
            self.path.append(cur)
            cur = parent[cur]
        self.path.append(self.start)

        self.path = self.path[::-1]
        
    def getLength(self):
        return len(self.path)
    
    def isEmpty(self):
        return not self.path
    
    def exportToFile(self, file):
        file.write('{}\n'.format(len(self.path)))
        for step in self.path:
            x, y = step
            file.write('({},{}) '.format(x, y))
        file.write('\n')
        
    def has(self, x, y):
        return (x, y) in self.path
        
    def showOnGUI(self, canvas, color = 'Path'):
        for step in self.path:
            x, y = step
            canvas.drawTile(color, x, y)
        canvas.drawStart(self.start)    
        canvas.drawGoal(self.goal)
        
    def show(self):
        print('[{}]'.format(len(self.path)), end=' ')
        for step in self.path:
            print(step, end=' ')

class Board:
    def __init__(self, view = None):
        self.width = self.height = 0
        self.start = self.goal = (0,0)
        self.grid = None
        self.view = view
    
    ### Getter
    def getWidth(self):
        return self.width
    
    def getHeight(self):
        return self.height
    
    def getStart(self):
        return self.start
    
    def getGoal(self):
        return self.goal
    
    def getCell(self, row, column):
        return self.grid[row][column]
    
    ### Set size
    def setSize(self, nwidth, nheight):
        self.width = nwidth
        self.height = nheight
        self.grid = [[0 for col in range(nwidth)] for row in range(nheight)]
    
    ### Set types of cell on board
    def setStart(self, x, y):
        if (self.isInBoard(x, y)):
            self.start = (x, y)

    def setGoal(self, x, y):
        if (self.isInBoard(x, y)):
            self.goal = (x, y)
    
    def toggleCell(self, x, y):
        if (x, y) == self.start or (x, y) == self.goal:
            return
        if (self.isInBoard(x, y)):
            self.grid[x][y] = 1 - self.grid[x][y]
            self.view.updateTile(x, y)
            
    ### Randomize board
    def randomOpenCell(self):
        x = random.randint(0, self.height - 1)
        y = random.randint(0, self.width - 1)
        while self.grid[x][y] == 1:
            x = random.randint(0, self.height - 1)
            y = random.randint(0, self.width - 1)
        return (x, y)

    def randomize(self):
        for x in range(self.height):
            for y in range(self.width):
                self.grid[x][y] = 1 if random.random() >= 0.8 else 0
        self.start = self.randomOpenCell()
        self.goal = self.randomOpenCell()

    ### Create the board from file
    def createBoardFromFile(self, fileDir):
        n, (xStart, yStart), (xGoal, yGoal), grid = self.getDataFromFile(fileDir)
        
        self.width = self.height = n
        self.grid = grid
        self.setStart(xStart, yStart)
        self.setGoal(xGoal, yGoal)
        
    def getDataFromFile(self, fileDir):
        f = open(fileDir, "r")

        n = int(f.readline())
        (xStart, yStart) = [int(n) for n in f.readline().split(' ')]
        (xGoal, yGoal) = [int(n) for n in f.readline().split(' ')]

        grid = []
        for line in f.read().splitlines():
            grid.append([int(n) for n in line.split(' ')[0:n]])
            
        f.close()
        return n, (xStart, yStart), (xGoal, yGoal), grid
    
    ### Export the solution to file
    def findPath(self, algoName, heuristic, Epsilon):
        algoGenerator = SearchAlgorithmFactory()
        algo = (algoGenerator.getAlgorithm(algoName))(self, heuristic, Epsilon)
        now = time.time()
        parent = algo.search()
        print('Searching takes {} s'.format((time.time() - now)))
        return Path(parent, self.start, self.goal)

    def exportSolutionToFile(self, fileDir):
        file = open(fileDir, 'w')
        self.saveSolutionToFile(file, euclid, 1)
    
    def saveSolutionToFile(self, file, heuristic, Epsilon = 1):
        path = self.findPath("AStar", heuristic, Epsilon)
        if path.isEmpty():
            file.write("-1")
            return
           
        path.exportToFile(file)
        
        n = len(self.grid)
        for row in range(n):
            for cell in range(n):
                if((row, cell) == self.start):
                    file.write('S')
                elif((row, cell) == self.goal):
                    file.write('G')
                elif(path.has(row, cell)):
                    file.write('x')
                else:
                    file.write('-' if self.grid[row][cell] == 0 else 'o')
                file.write(" ")
            file.write('\n')
        
        return
    
    ### Save current board to file
    def saveToFile(self, file):
        n = len(self.grid)
        file.write('{}\n'.format(n))
        file.write('{} {}\n'.format(self.start[0], self.start[1]))
        file.write('{} {}\n'.format(self.goal[0], self.goal[1]))
        for row in range(n):
            for column in range(n):
                file.write('{} '.format(self.grid[row][column]))
            file.write('\n')
    
    ### Represent the board
    def show(self):
        n = len(self.grid)
        for row in range(n):
            for cell in range(n):
                if((row, cell) == self.start):
                    print('S', end = " ")
                elif((row, cell) == self.goal):
                    print('G', end = " ")
                else:
                    print('-' if self.grid[row][cell] == 0 else 'o', end = " ")
            print()
        
    def showOnGUI(self):
        self.view.drawBoard()
        self.view.drawStart(self.start)
        self.view.drawGoal(self.goal)
    
    ### Check if point in board
    def isInBoard(self, x, y):
        return 0 <= x < self.height and 0 <= y < self.width
    
    ### Get all neighbors of current cell
    def getNeighbors(self, cell):
        (x, y) = cell
        if not self.isInBoard(x, y): return []
        
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]

        neighbors = []
        for i in range(8):
            nx,ny = x + dx[i], y + dy[i]
            if (self.isInBoard(nx,ny) and self.grid[nx][ny] == 0):
                neighbors.append((nx,ny))
                
        return neighbors
    
    ### Visualize
    def visualize(self, heuristic, Epsilon = 1, sleepTime = 0.2):
        open_set = PriorityQueue()
        close_set = []
        cost = {}
        parent = {}
        
        open_set.push((0, self.start))
        cost[self.start] = 0
        parent[self.start] = None
        
        while not open_set.isEmpty():
            curCost, curCell = open_set.pop()
            
            if curCell == self.goal:
                break
                
            close_set.append(curCell)
            self.view.drawTile('Visited', curCell[0], curCell[1])
            time.sleep(sleepTime)
            
            for nextCell in self.getNeighbors(curCell):
                if nextCell not in close_set:
                    g = cost[curCell] + 1
                    h = Epsilon * heuristic(nextCell, self.goal)
                    f = g + h
                    if nextCell not in cost or cost[nextCell] > g:
                        self.view.updateTextTile(nextCell[0], nextCell[1], '{0:.1f}'.format(f))
                        self.view.drawTile('Visiting', nextCell[0], nextCell[1])
                        
                        cost[nextCell] = g
                        open_set.push((f, nextCell))
                        parent[nextCell] = curCell
            time.sleep(sleepTime)
                        
        path = Path(parent, self.start, self.goal)
        path.showOnGUI(self.view)
    
    # ARA visualize
    def key(self, s, cost, Epsilon, heuristic):
        return cost[s] + Epsilon * heuristic(s, self.goal)
        
    def ImprovePath(self, open_set, close_set, incons_set, cost, parent, Epsilon, heuristic):
        while not open_set.isEmpty() and self.key(self.goal, cost, Epsilon, heuristic) > open_set.peek()[0]:
            curCost, curCell = open_set.pop()
            close_set.append(curCell)
            
            self.view.drawTile('Visited', curCell[0], curCell[1])
            time.sleep(0.001)
            
            for nextCell in self.getNeighbors(curCell):
                if nextCell not in cost or cost[nextCell] > cost[curCell] + 1:  
                    cost[nextCell] = cost[curCell] + 1
                    parent[nextCell] = curCell
                    if nextCell not in close_set:
                        f = self.key(nextCell, cost, Epsilon, heuristic)
                        self.view.updateTextTile(nextCell[0], nextCell[1], '{0:.1f}'.format(f))
                        self.view.drawTile('Visiting', nextCell[0], nextCell[1])
                        open_set.push((f, nextCell))
                    else:
                        incons_set.append(nextCell)

            time.sleep(0.001)
    
    
    
    def MainSearch(self, heuristic, startingEpsilon, timeLimit):
        if (timeLimit == 0):
            timeLimit = 1e18

        open_set = PriorityQueue()
        close_set = []
        incons_set = []
        
        cost = {}
        parent = {}
        
        Epsilon = startingEpsilon
        
        cost[self.start] = 0
        parent[self.start] = None
        cost[self.goal] = 1e9
        open_set.push((self.key(self.start, cost, Epsilon, heuristic), self.start))
        
        path = None
        now = time.time()
        time_max = now + timeLimit
        while Epsilon >= 1 and time.time() < time_max:
            self.ImprovePath(open_set, close_set, incons_set, cost, parent, Epsilon, heuristic)
            
            if time.time() < time_max:
                path = Path(parent, self.start, self.goal)
                path.showOnGUI(self.view, color = '#718093')
                print('Epsilon = {}'.format(Epsilon))
                path.show()
                print()
            else:
                late_path = Path(parent, self.start, self.goal)
                late_path.showOnGUI(self.view, color = 'red')
                break
            
            Epsilon -= 0.03125
            
            common_set = set()
            while not open_set.isEmpty():
                common_set.add(open_set.pop()[1])
            for cell in incons_set:
                common_set.add(cell)
            for cell in common_set:
                f = self.key(cell, cost, Epsilon, heuristic)
                open_set.push((f, cell))
                self.view.updateTextTile(cell[0], cell[1], '{0:.1f}'.format(f))
                
            incons_set = []
            close_set = []

        if path:
            path.showOnGUI(self.view)

class View(object):
    def __init__(self, canvas):
        self.canvas = canvas
        
        self.sizeSquare = 50
        self.fontSize = 10
        self.color = {
            'Start': '#e74c3c',
            'Goal': '#fdcb6e',
            'Empty': '#1abc9c',
            'Obstacle': '#2c3e50',
            'Visiting': '#ecf0f1',
            'Path': '#0984e3',
            'Visited': '#81ecec'
        }
        
        self.tiles = None
        self.scores = None
        
    def setBoard(self, grid):
        self.grid = grid

    def drawBoard(self):
        boardWidth = self.grid.getWidth()
        boardHeight = self.grid.getHeight()
        
        if (50 * boardHeight > 800):
            self.sizeSquare = 800 // boardHeight
        elif (50 * boardWidth > 1000):
            self.sizeSquare = 1000 // boardWidth
        else:
            self.sizeSquare = 50
            
        size = self.sizeSquare
        self.fontSize = self.sizeSquare // 4
        
        self.canvas.configure(width = size * boardWidth, height = size * boardHeight)
        
        characterToTileType = { 0: 'Empty', 1: 'Obstacle'}
        self.tiles = [[] for row in range(boardHeight)]
        self.scores = [[] for row in range(boardHeight)]
        for row in range(boardHeight):
            for column in range(boardWidth):
                self.tiles[row].append(self.newTile(row, column))
                self.scores[row].append(self.newTextTile(row, column))
                
    def eventTileClick(self, event, row, column):
        self.grid.toggleCell(row, column)

    def drawTile(self, tileType, x, y):
        size = self.sizeSquare
        x1 = y * size
        x2 = x1 + size
        y1 = x * size
        y2 = y1 + size
            
        if tileType in ['Start', 'Goal', 'Empty', 'Obstacle', 'Visiting', 'Visited', 'Path', 'Inconsistent']:
            self.canvas.itemconfigure(self.tiles[x][y], fill = self.color[tileType])
        else:
            self.canvas.itemconfigure(self.tiles[x][y], fill = tileType)
        
        self.canvas.update()
            
    def drawStart(self, pos):
        row, column = pos
        self.drawTile('Start', row, column)
        
    def drawGoal(self, pos):
        row, column = pos
        self.drawTile('Goal', row, column)
    
    def newTile(self, row, column):
        size = self.sizeSquare
        x1 = column * size
        y1 = row * size
        x2 = x1 + size
        y2 = y1 + size
        
        characterToTileType = { 0: 'Empty', 1: 'Obstacle'}
        tileType = characterToTileType[self.grid.getCell(row, column)]
        rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill = self.color[tileType], outline = 'white')
        self.canvas.tag_bind(rect, '<Button-1>', lambda event, row = row, column = column: self.eventTileClick(event, row, column))
        return rect
        
    def updateTile(self, row, column):
        characterToTileType = { 0: 'Empty', 1: 'Obstacle'}
        tileType = characterToTileType[self.grid.getCell(row, column)]
        self.canvas.itemconfigure(self.tiles[row][column], fill = self.color[tileType])
        return
        
    def newTextTile(self, row, column):
        size = self.sizeSquare
        x = column * size + size // 2
        y = row * size + size // 2
        
        texttile = self.canvas.create_text(x, y, text = '', font = (None, self.fontSize))
        return texttile
    
    def updateTextTile(self, row, column, text):
        self.canvas.itemconfigure(self.scores[row][column], text = text)

def manhattan(cur, goal):
        return abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
    
def euclid(cur, goal):
        return math.sqrt((cur[0] - goal[0])**2 + (cur[1] - goal[1])**2)
    
def diagonal(cur, goal):
        return max(abs(cur[0] - goal[0]), abs(cur[1] - goal[1]))

class GUI:
    def __init__(self):
        self.window = Tk()
        self.window.configure(background='#2d3436')
        self.window.title("Find your own path in life")
        self.initializeComponents()

    def initializeComponents(self):
        self.canvas = Canvas(self.window, width = 0, height = 0, bg = "white", highlightthickness=0)
        self.view = View(self.canvas)

        self.board = Board(self.view)
        self.view.setBoard(self.board)

        self.board.setSize(10, 10)
        self.board.randomize()
        self.board.showOnGUI() 
        
        self.new_button = Button(self.window, text="New", bg="#3498db", fg="white", highlightthickness = 0,
                                 borderwidth = 0, command=lambda: self.askNewSize())
        self.save_button = Button(self.window, text="Save", bg="#3498db", fg="white", highlightthickness = 0,
                                  borderwidth = 0, command=lambda: self.askSaveFile())
        self.load_button = Button(self.window, text="Load", bg="#3498db", fg="white", highlightthickness = 0,
                                  borderwidth = 0, command=lambda: self.loadFile())
        self.reset_button = Button(self.window, text="Reset", bg="#3498db", fg="white", highlightthickness = 0,
                                   borderwidth = 0, command=lambda: self.reset())
        
        self.idHeuristic = IntVar()
        self.idHeuristic.set(1)
        self.manhattan_rdbtn = Radiobutton(self.window, text = "Manhattan", bg = "#2d3436", fg = "white", 
                                           selectcolor='black', highlightthickness = 0,  variable = self.idHeuristic, value = 0)
        self.euclid_rdbtn = Radiobutton(self.window, text = "Euclid", bg = "#2d3436",  fg = "white", 
                                           selectcolor='black',  highlightthickness = 0, variable = self.idHeuristic, value = 1)
        self.diagonal_rdbtn = Radiobutton(self.window, text = "Diagonal", bg = "#2d3436",  fg = "white", 
                                           selectcolor='black', highlightthickness = 0, variable = self.idHeuristic, value = 2)
        
        self.Epsilon = 1
        self.valueEpsilon = Label(self.window, text="Epsilon: 1.0", fg="white", bg="#2d3436")
        self.epsilon_button = Button(self.window, text="Set Epsilon", bg="#ff7675", fg="white", highlightthickness = 0,
                                     borderwidth = 0, command=lambda: self.askEpsilon())
        
        self.Time = 0
        self.valueTime = Label(self.window, text="Time: oo", fg="white", bg="#2d3436")
        self.time_button = Button(self.window, text="Set Time", bg="#ff7675", fg="white", highlightthickness = 0,
                                     borderwidth = 0, command=lambda: self.askTime())
        
        self.solve_button = Button(self.window, text="Solve", bg="#ff7675", fg="white", highlightthickness = 0,
                                   borderwidth = 0, command=lambda: self.startSearching())
        self.export_button = Button(self.window, text="Export", bg="#ff7675", fg="white", highlightthickness = 0,
                                    borderwidth = 0, command=lambda: self.exportFile())
    def main_menu(self):
        
        # ===== Canvas =====
        self.canvas.pack(side = RIGHT, anchor=E, padx = 10, pady = 10, expand = True)

        # ===== New button =====
        self.new_button.config(font=('times', 12, 'bold'), width=10)
        self.new_button.pack(side = TOP, anchor=W, padx = 20, pady = (20, 3))
        
        # ===== Save button =====
        self.save_button.config(font=('times', 12, 'bold'), width=10)
        self.save_button.pack(side = TOP, anchor=W, padx = 20, pady = 3)
       
         # ===== load button =====
        self.load_button.config(font=('times', 12, 'bold'), width=10)
        self.load_button.pack(side = TOP, anchor=W, padx = 20, pady = 3)
        
        # ===== Reset button =====
        self.reset_button.config(font=('times', 12, 'bold'), width=10)
        self.reset_button.pack(side = TOP, anchor=W, padx = 20, pady = (3, 20))
        
        # ===== Heuristic radio buttons =====
        self.manhattan_rdbtn.pack(side = TOP, anchor = SW, padx = 20, pady = 2)
        self.euclid_rdbtn.pack(side = TOP, anchor = SW, padx = 20, pady = 2)
        self.diagonal_rdbtn.pack(side = TOP, anchor = SW, padx = 20, pady = 2)
        
        # ===== Set Epsilon button =====
        self.valueEpsilon.pack(side = TOP, anchor=SW, padx = 20, pady = (20, 3))
        self.epsilon_button.config(font=('times', 12, 'bold'), width=10)
        self.epsilon_button.pack(side = TOP, anchor=SW, padx = 20, pady = 3)
        
        # ===== Set Time button =====
        self.valueTime.pack(side = TOP, anchor=SW, padx = 20, pady = (10, 3))
        self.time_button.config(font=('times', 12, 'bold'), width=10)
        self.time_button.pack(side = TOP, anchor=SW, padx = 20, pady = 3)
        
        # ===== Solve button =====
        self.solve_button.config(font=('times', 12, 'bold'), width=10)
        self.solve_button.pack(side = TOP, anchor=SW, padx = 20, pady = (20, 3))
        
        # ===== Export button =====
        self.export_button.config(font=('times', 12, 'bold'), width=10)
        self.export_button.pack(side = TOP, anchor=SW, padx = 20, pady = 3)
    
        self.window.mainloop()
        
    def askNewSize(self):
        try:
            var = IntVar()
            var.set(simpledialog.askinteger("Create your new size", "How big do you want your life?",
                                     parent=self.window,
                                     minvalue=2, maxvalue=100))
            new_size = var.get()
            self.board.setSize(new_size, new_size)
            self.board.setStart(0, 0)
            self.board.setGoal(new_size - 1, new_size - 1)
            self.board.showOnGUI()
        except:
            messagebox.showwarning("Warning", "Don't worry, your value cannot be that small.")
        
    def askEpsilon(self):
        try:
            var = DoubleVar()
            var.set(simpledialog.askfloat("Set your new pace", "How fast do you want to live?",
                                     parent=self.window,
                                     minvalue=1, maxvalue=100))
            self.Epsilon = var.get()
            temp = "Epsilon: " + str(self.Epsilon)
            self.valueEpsilon.config(text = temp)
        except:
            messagebox.showwarning("Warning", "To live is to move. Don't stop.")
    
    def askTime(self):
        try:
            var = DoubleVar()
            var.set(simpledialog.askfloat("Set your new pace", "what time do you want to live?",
                                     parent=self.window,
                                     minvalue=0, maxvalue=100))
            self.Time = var.get()
            temp = "Time: " + (str(self.Time) if self.Time > 0 else 'oo')
            self.valueTime.config(text = temp)
        except:
            messagebox.showwarning("Warning", "To live is to sleep. Don't wake up.")
        
    
    def reset(self):
        try:
            self.board.showOnGUI()
        except:
            messagebox.showerror("Error", "Empty lives do not need resetting.")
            
    def startDrawingFromFile(self, path):
        self.board.createBoardFromFile(path)
        self.board.showOnGUI()
        
    def loadFile(self):
        try:
            var = StringVar()
            file = filedialog.askopenfilename()
            if not file: return
            var.set(file)
            path = var.get()
            self.startDrawingFromFile(path)
        except:
            messagebox.showerror("Error", "Not only is your life empty, it failed to load.")
        
    def getHeuristic(self):
        heuristics = [ manhattan, euclid, diagonal ]
        return heuristics[self.idHeuristic.get()]

    def startSearching(self):
            searchHeuristic = self.getHeuristic()
            
            self.setStateListButtons([self.reset_button, self.load_button, self.new_button, self.solve_button], 'disabled')

            self.board.showOnGUI()
            self.board.MainSearch(searchHeuristic, self.Epsilon, self.Time)

            self.setStateListButtons([self.reset_button, self.load_button, self.new_button, self.solve_button], 'normal')
    
    def exportFile(self):
        try:
            file = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
            if not file: return
            
            searchHeuristic = self.getHeuristic()
            self.board.saveSolutionToFile(file, searchHeuristic, self.Epsilon)
            file.close()
        except:
            messagebox.showerror("Error", "Cannot solve your empty life.")

    def askSaveFile(self):
        try:
            file = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
            if not file: return
            self.board.saveToFile(file)
            file.close()
        except:
            messagebox.showerror("Error", "Cannot save your empty life.")
            
    def setStateListButtons(self, buttons, btnstate):
        for button in buttons:
            button.config(state = btnstate)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run A* with input_dir and save solution to output_dir. If no directory is provided, the GUI will show up.')
    parser.add_argument('input_dir', type=str, help='Input directory', nargs='?', default = None)
    parser.add_argument('output_dir', type=str, help='Output directory', nargs='?', default = None)
    args = parser.parse_args()

    if args.input_dir is None:
        try:
            app = GUI()
            app.main_menu()
        except:
            messagebox.showerror("Error", "Why do you end your life so soon?")
    else: 
        if args.output_dir is None:
            args.output_dir = 'solution.txt'
        board = Board()
        board.createBoardFromFile(args.input_dir)
        board.exportSolutionToFile(args.output_dir)
        print('Successfully export solution of {} to {}'.format(args.input_dir, args.output_dir))
