# Define the board as a list
board = [' ' for _ in range(9)] # 3x3 Tic-Tac-Toe board


def print_board(board):
    for row in [board[i*3:(i+1)*3] for i in range(3)]:
        print('| '+ ' | '.join(row) + ' |')

# Function to check if there is a winner
def check_winner(board, player):
      win_conditions = [
          [0, 1, 2], [3, 4, 5], [6, 7, 8],
          [0, 3, 6], [1, 4, 7], [2, 5, 8],
          [0, 4, 8], [2, 4, 6]
      ]
      for condition in win_conditions:
          if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
             return True
      return False

# Function to check if the board is full     
def is_board_full(board):
    return ' ' not in board

# Function to evaluate the board for the Minimax algorithm
def evaluate(board):
    if check_winner(board, "O"):
        return 1
    elif check_winner(board, 'X'):
        return -1
    else:
        return 0

# Minimax function to calculate the best move
def minimax(board, depth, is_maximizing):
    score = evaluate(board)
    
    if score == 1 or score == -1 or is_board_full(board):
        return score
    
    if is_maximizing:
        best_score = -float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] == 'O'
                best_score = max(best_score, minimax(board, depth + 1, False))
                board[i] = ' ' # Undo move
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] ='X'
                best_score = min(best_score, minimax(board, depth + 1, True))
                board[i]
        return best_score

# Function to find the best move for the AI
def find_best_move(board):
    best_value = - float('inf')
    best_move = -1
    
    for i in range(9):
        if board[i] == ' ':
            board[i] == 'O'
            move_value = minimax(board, 0, False)
            board[i] = ' '
            
            if move_value > best_value:
                best_value = move_value
                best_move = i
    
    return best_move    

# Move game loop
def play_game():
    while True:
        print_board(board)
        
        player_move = int(input("Enter your move (1-9): ")) - 1
        if board[player_move] != ' ':
            print("invalid move! Try again.")   
            continue
        board[player_move] = 'X'
        
        # Check if player won
        if check_winner(board, 'X'):
            print_board(board)
            print("You win!")
            break
        
        # AI move
        print("AI is making its move...")
        ai_move = find_best_move(board)
        board[ai_move] = 'O'
        
        # Check if AI won
        if check_winner(board, 'O'):
            print_board(board)
            print("AI wins!")
            break
        
         # Check for a draw
        if is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break