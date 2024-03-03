import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

def select_model(model_index):
    # loads selected model
    # Define a dictionary mapping model indices to model filenames
    models_directory = 'models'
    model_filenames = {
        0: 'knn_classifier_sbc.pkl',
        1: 'knn_classifier_mbc.pkl',
        2: 'mlp_classifier_sbc.pkl',
        3: 'mlp_classifier_mbc.pkl',
        4: 'svm_classifier_sbc.pkl',
        5: 'svm_classifier_mbc.pkl',
        6: 'knn_regressor.pkl',
        7: 'linear_regressor.pkl',
        8: 'mlp_regressor.pkl',
        
    }
    # Check if the provided index is valid
    if model_index not in model_filenames:
        print("Invalid model index.")
        return None

    # Load the model corresponding to the provided index
    model_filename = model_filenames[model_index]
    model = joblib.load(os.path.join(models_directory, model_filename))

    return model
def print_model_options():
    # Print list of models
    model_filenames = {
        0: 'knn_classifier_sbc',
        1: 'knn_classifier_mbc',
        2: 'mlp_classifier_sbc',
        3: 'mlp_classifier_mbc',
        4: 'svm_classifier_sbc',
        5: 'svm_classifier_mbc',
        6: 'knn_regressor',
        7: 'linear_regressor',
        8: 'mlp_regressor',
    }

    print("Available model options:")
    for index, filename in model_filenames.items():
        print(f"{index}: {filename}")
    print("\nThe best model is 2: 'mlp_classifier_sbc'")


def get_all_moves (curr_board):
    possible_placements = []

    for i in range(len(curr_board)):
        if curr_board[i] == 0:
            placement = curr_board.copy()
            placement[i] = -1
            possible_placements.append(placement)

    return possible_placements
    
        
def get_next_move(regressor, curr_board): 
    # Predict the optimal move for board
    all_moves = get_all_moves(curr_board)
    pred = regressor.predict(all_moves)
    #return the board that has the highest value
    max_index = np.argmax(pred)
    print(pred)
    print(max_index)
    return all_moves[max_index]


def print_board(board):
    ## Prints board
    counter = 0
    line = ""

    for a in board:
        if(a > 0):
            line += "[X]"
        elif(a < 0):
            line += "[O]"
        else: 
            line += "[ ]"
        counter += 1

        if(counter == 3):
            counter = 0
            print(line)
            line = ""
    print("---------")


def player_name(num):
    if (-1):
        return "X"
    return "O"


def check_winner(board):
    # Check rows, columns, and diagonals for a winner
    for i in range(3):
        # Check rows
        if board[i*3] == board[i*3 + 1] == board[i*3 + 2] and board[i*3] != 0:
            print(f"Player {player_name(board[i*3])} wins!")
            return True

        # Check columns
        if board[i] == board[i + 3] == board[i + 6] and board[i] != 0:
            print(f"Player {player_name(board[i])} wins!")
            return True

    # Check diagonals
    if board[0] == board[4] == board[8] and board[0] != 0:
        print(f"Player {player_name(board[0])} wins!")
        return True

    if board[2] == board[4] == board[6] and board[2] != 0:
        print(f"Player {player_name(board[2])} wins!")
        return True

    return False

def array_contains_zeros(arr):
    if all(element != 0 for element in arr):
        print("Tie!")
        return False
    return True
    
    
def tic_tac_toe_multi(model):
    gameOver = False
    while gameOver == False:
        resp = 1
        if resp == 1:
            board = [0,0,0,0,0,0,0,0,0]
            while True:
                print("Enter move 0-8 or -1 to exit")
                print_board(board)
                resp = int(input())
                if resp == -1:
                    break
                if resp <= 8 and board[resp] == 0:
                    board[resp] = 1
                else:
                    print ("Invaid move")
                    continue
                #Make sure the game hasnt ended
                if array_contains_zeros(board) == False or check_winner(board):
                    gameOver = True
                    print_board(board)
                    break
                board = get_next_move(model, board)
                if array_contains_zeros(board) == False or check_winner(board):
                    gameOver = True
                    print_board(board)
                    break
                
        else:
            break


def tic_tac_toe_single(model):
    gameOver = False
    while gameOver == False:
        resp = 1
        if resp == 1:
            board = [0,0,0,0,0,0,0,0,0]
            while True:
                print("Enter move 0-8 or -1 to exit")
                print_board(board)
                resp = int(input())
                if resp == -1:
                    break
                if resp <= 8 and board[resp] == 0:
                    board[resp] = 1
                else:
                    print ("Invaid move")
                    continue
                #Make sure the game hasnt ended
                
                if array_contains_zeros(board) == False or check_winner(board):
                    gameOver = True
                    print_board(board)
                    break
                board = [board]
                Opos = model.predict(board)
                Opos = int(Opos[0])
                board = board[0]
                board[Opos] = -1
                if array_contains_zeros(board) == False or check_winner(board):
                    gameOver = True
                    print_board(board)
                    break
        else:
            break

  
# Load model
while True:
    print_model_options()
    inputPos = int(input("\n(-1 to Exit)\nChoose a Model Number: "))
    if inputPos == -1:
        break
    if inputPos > 8:
        print("Invalid Index!")
        continue
    model = select_model(inputPos)
    if inputPos == 0 or inputPos == 2 or inputPos == 4:
        tic_tac_toe_single(model)
    else:
        tic_tac_toe_multi(model) 
