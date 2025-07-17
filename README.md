# Pacman Game

A simple Pacman game built with Pygame!

## Features

- **Pacman Character**: Yellow circle that you control
- **Ghosts**: 4 colored ghosts (Red, Pink, Cyan, Orange) with simple AI
- **Dots**: Collect white dots to score points
- **Maze**: Randomly generated maze with walls
- **Scoring**: Earn 10 points per dot collected
- **Lives**: Start with 3 lives, lose one when touching a ghost
- **Win/Lose Conditions**: Win by collecting all dots, lose by running out of lives

## Controls

- **Arrow Keys**: Move Pacman (Up, Down, Left, Right)
- **R**: Restart game (when game over or won)
- **ESC**: Quit game

## Installation

1. Make sure you have Python installed (3.7 or higher)
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Run

Navigate to the games directory and run:
```
python first.py
```

## Game Rules

1. Use arrow keys to move Pacman around the maze
2. Collect all the white dots to win
3. Avoid the colored ghosts - they will chase you!
4. You have 3 lives - each time you touch a ghost, you lose a life
5. The game wraps around the screen edges
6. When you lose a life, you and the ghosts return to starting positions

## Game Elements

- **Blue rectangles**: Walls (cannot pass through)
- **Yellow circle**: Pacman (you)
- **Colored circles**: Ghosts (avoid these!)
- **White dots**: Collectible points
- **UI**: Shows score, lives remaining, and dots left

Enjoy playing! 