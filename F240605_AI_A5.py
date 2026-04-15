import copy
import os
from collections import deque

class SudokuCSP:
    def __init__(self, board):
        self.board = copy.deepcopy(board) # 9x9 grid (0 = unassigned)
        self.domains = {}
        
        # Statistics
        self.backtrack_calls = 0
        self.backtrack_failures = 0
        
        self._init_domains()

    def _init_domains(self):
        """Assign initial candidate sets to every cell."""
        for i in range(9):
            for j in range(9):
                val = self.board[i][j]
                self.domains[(i, j)] = {val} if val != 0 else set(range(1, 10))

    @staticmethod
    def get_neighbors(row: int, col: int) -> set:
        """
        Return every cell that shares a row, column, or 3x3 box with
        (row, col). These cells participate in the all-different constraint.
        """
        nb = set()
        for j in range(9):                    # same row
            if j != col:
                nb.add((row, j))
        for i in range(9):                    # same column
            if i != row:
                nb.add((i, col))
        br, bc = 3 * (row // 3), 3 * (col // 3)
        for i in range(br, br + 3):           # same 3x3 box
            for j in range(bc, bc + 3):
                if (i, j) != (row, col):
                    nb.add((i, j))
        return nb

    def ac3(self) -> bool:
        """Enforce arc consistency globally before any search begins."""
        queue = deque()
        for i in range(9):
            for j in range(9):
                for nb in self.get_neighbors(i, j):
                    queue.append(((i, j), nb))

        while queue:
            xi, xj = queue.popleft()
            if self._revise(xi, xj):
                if not self.domains[xi]:
                    return False      # domain empty => no solution
                # Re-check all arcs pointing INTO xi
                for xk in self.get_neighbors(xi[0], xi[1]):
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def _revise(self, xi, xj) -> bool:
        """Remove unsupported values from D(xi)."""
        revised = False
        for val in set(self.domains[xi]):
            if all(v == val for v in self.domains[xj]):  # no support
                self.domains[xi].discard(val)
                revised = True
        return revised

    def _forward_check(self, row: int, col: int, val: int):
        """
        After assigning val to (row, col), propagate: remove val from every
        unassigned neighbour's domain. Returns pruned dictionary or None on failure.
        """
        pruned = {}
        for nb in self.get_neighbors(row, col):
            ni, nj = nb
            if self.board[ni][nj] == 0 and val in self.domains[nb]:
                pruned.setdefault(nb, set()).add(val)
                self.domains[nb].discard(val)
                if not self.domains[nb]:          # domain wiped out
                    self._restore_domains(pruned) # undo before signalling failure
                    return None
        return pruned

    def _restore_domains(self, pruned: dict):
        """Undo the domain reductions recorded in pruned."""
        for cell, removed_vals in pruned.items():
            self.domains[cell].update(removed_vals)

    def _select_unassigned_variable(self):
        """Minimum Remaining Values (MRV) heuristic."""
        best, min_size = None, float('inf')
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    d = len(self.domains[(i, j)])
                    if d < min_size:
                        min_size, best = d, (i, j)
        return best

    def _is_consistent(self, row: int, col: int, val: int) -> bool:
        """True iff placing val at (row, col) does not conflict with any neighbour."""
        for nb in self.get_neighbors(row, col):
            if self.board[nb[0]][nb[1]] == val:
                return False
        return True

    def _backtrack(self) -> bool:
        """Recursive backtracking search with forward checking and MRV."""
        self.backtrack_calls += 1

        var = self._select_unassigned_variable()
        if var is None:
            return True         # every cell filled => solution found

        row, col = var

        for val in sorted(self.domains[(row, col)]):   # sorted for reproducibility
            if self._is_consistent(row, col, val):

                # assign
                self.board[row][col] = val

                # propagate via forward checking
                pruned = self._forward_check(row, col, val)

                if pruned is not None:              # no immediate contradiction
                    if self._backtrack():
                        return True                 # solution found downstream

                # undo assignment + domain changes
                self.board[row][col] = 0
                if pruned is not None:
                    self._restore_domains(pruned)

        # All values exhausted for this variable => backtrack
        self.backtrack_failures += 1
        return False

    def solve(self):
        """
        Main solve function wrapping AC-3 and Backtracking.
        Returns the solution as an assignment dictionary format to fit the assignment reqs.
        """
        self.backtrack_calls = 0
        self.backtrack_failures = 0

        # Step 1: Enforce Arc Consistency
        if not self.ac3():
            return None # Unsolvable from the start

        # Step 2: Begin Backtracking Search
        if self._backtrack():
            # Step 3: Convert the solved 2D board back into the requested dictionary format
            assignment = {}
            for r in range(9):
                for c in range(9):
                    assignment[(r, c)] = self.board[r][c]
            return assignment
        
        return None

def read_board(file_path):
    """Read a 9x9 Sudoku board from a file."""
    if not os.path.exists(file_path):
        return None
    board = []
    with open(file_path, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                board.append([int(char) for char in clean_line])
    return board

def format_board(assignment):
    """Format the completed assignment dictionary into the requested string format."""
    lines = []
    for r in range(9):
        row_vals = []
        for c in range(9):
            row_vals.append(str(assignment.get((r, c), 0)))
        
        # Formatting with pipes
        formatted_row = f"{row_vals[0]} {row_vals[1]} {row_vals[2]} | {row_vals[3]} {row_vals[4]} {row_vals[5]} | {row_vals[6]} {row_vals[7]} {row_vals[8]}"
        lines.append(formatted_row)
        
        # Adding horizontal dividers
        if r == 2 or r == 5:
            lines.append("- - - + - - - + - - -")
    
    return "\n".join(lines)


def main():
    files = ["easy.txt", "medium.txt", "hard.txt", "veryhard.txt"]
    
    for filename in files:
        board = read_board(filename)
        if not board:
            print(f"File {filename} not found. Skipping...\n")
            continue
            
        print(f"========================================")
        print(f"Solving {filename}...")
        print(f"========================================\n")
        
        solver = SudokuCSP(board)
        solution = solver.solve()
        
        out_filename = filename.replace(".txt", "_output.txt")
        
        with open(out_filename, 'w') as f:
            if solution:
                # Construct the output string without the analysis text
                output_text = (
                    "Solved Sudoku:\n" +
                    format_board(solution) + "\n\n" +
                    f"Backtrack Calls: {solver.backtrack_calls}\n" +
                    f"Backtrack Failures: {solver.backtrack_failures}\n"
                )
                
                # Print exactly as requested
                print(output_text)
                
                # Write it to the file
                f.write(output_text)
            else:
                failure_msg = "No solution exists for this board.\n"
                print(failure_msg)
                f.write(failure_msg)

if __name__ == "__main__":
    main()