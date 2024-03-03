import numpy as np

def initBoard():
    board = [0] * 9 #0 = blank, 1 = X, -1 = O
    printBoard(board)
    return board

def placeMark(board, val, ind):
    if(board[ind] == 0):
        board[ind] = val
    else:
        print("Invalid Mark") #this doesn't need to be handled here but it wouldn't hurt

    if(checkWinner(board) != 0 or val == -1): #only need to display board if the AI just went, so the player can react
        printBoard(board)


def printBoard(board):
    if(handleWin(board)):
        print("Board State:")
    else:
        print("Your Turn:")
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

def checkWinner(board): #return value of winning symbol or 0 if no winner
    if(board[0] == board[1] == board[2] != 0):
        return board[0]
    elif(board[3] == board[4] == board[5] != 0):
        return board[3]
    elif(board[6] == board[7] == board[8] != 0):
        return board[6]
    elif(board[0] == board[3] == board[6] != 0):
        return board[0]
    elif(board[1] == board[4] == board[7] != 0):
        return board[1]
    elif(board[2] == board[5] == board[8] != 0):
        return board[2]
    elif(board[0] == board[4] == board[8] != 0):
        return board[0]
    elif(board[2] == board[4] == board[6] != 0):
        return board[2]
    else:
        return 0

def handleWin(board):
    val = checkWinner(board)

    if(val != 0):
        print(("The Computer" if val == -1 else "The Player") + " wins!") 
        return True
    else:
        return False


game = initBoard()
placeMark(game, 1, int(input("Enter position: ")))
placeMark(game, -1, 6) #ai

placeMark(game, 1, int(input("Enter position: ")))
placeMark(game, -1, 4) #ai

placeMark(game, 1, int(input("Enter position: "))) # this would go until someone wins0
