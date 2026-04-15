# Chomp Game with Gemini AI

This is an implementation of the Chomp game where you can play against Google's Gemini AI. The game is played on a grid where players take turns removing squares, and all squares to the right and below the chosen square are also removed. The player who is forced to take the poison square (top-left corner) loses.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

You can get a Gemini API key from the [Google AI Studio](https://makersuite.google.com/app/apikey).

## How to Play

1. Run the game:
```bash
python chomp_game.py
```

2. The game is played on a 5x5 grid. On your turn:
   - Enter the row number (0-4)
   - Enter the column number (0-4)
   - The square you choose and all squares to its right and below will be removed

3. The game continues until one player is forced to take the poison square (top-left corner).

## Game Rules

- Players take turns removing squares from the board
- When a square is removed, all squares to its right and below are also removed
- The poison square (2) is in the top-left corner
- The player who is forced to take the poison square loses

## Implementation Details

The game is implemented using an object-oriented approach with three main classes:
- `ChompBoard`: Manages the game board and move validation
- `GeminiPlayer`: Handles the AI player using the Gemini API
- `ChompGame`: Coordinates the game flow and player turns