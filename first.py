import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CELL_SIZE = 20
MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

# Colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PINK = (255, 182, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)

# Game states
PLAYING = 0
GAME_OVER = 1
WIN = 2

class Pacman:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = [0, 0]
        self.next_direction = [0, 0]
        self.score = 0
        self.lives = 3
        self.speed_counter = 0
        self.speed_delay = 8  # Higher number = slower movement
        
    def move(self, maze):
        self.speed_counter += 1
        if self.speed_counter < self.speed_delay:
            return
            
        self.speed_counter = 0
        
        # Update direction if possible
        if self.can_move(maze, self.next_direction):
            self.direction = self.next_direction.copy()
        
        # Move in current direction
        new_x = self.x + self.direction[0]
        new_y = self.y + self.direction[1]
        
        if self.can_move(maze, self.direction):
            self.x = new_x
            self.y = new_y
            
            # Wrap around screen edges
            if self.x < 0:
                self.x = MAZE_WIDTH - 1
            elif self.x >= MAZE_WIDTH:
                self.x = 0
                
    def can_move(self, maze, direction):
        new_x = self.x + direction[0]
        new_y = self.y + direction[1]
        
        # Check bounds
        if new_x < 0 or new_x >= MAZE_WIDTH or new_y < 0 or new_y >= MAZE_HEIGHT:
            return True  # Allow wrapping
            
        # Check if wall
        return maze[new_y][new_x] != 1
        
    def set_direction(self, direction):
        self.next_direction = direction
        
    def draw(self, screen):
        # Draw Pacman as a yellow circle
        center_x = self.x * CELL_SIZE + CELL_SIZE // 2
        center_y = self.y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 2
        
        pygame.draw.circle(screen, YELLOW, (center_x, center_y), radius)

class Ghost:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.direction = [0, 0]
        self.speed_counter = 0
        self.speed_delay = 12  # Ghosts move slower than Pacman
        
    def move(self, maze):
        self.speed_counter += 1
        if self.speed_counter < self.speed_delay:
            return
            
        self.speed_counter = 0
        
        # Simple AI: randomly change direction or continue
        if random.random() < 0.05 or not self.can_move(maze, self.direction):  # Reduced random direction change
            possible_directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
            random.shuffle(possible_directions)
            
            for direction in possible_directions:
                if self.can_move(maze, direction):
                    self.direction = direction
                    break
        
        # Move in current direction
        if self.can_move(maze, self.direction):
            self.x += self.direction[0]
            self.y += self.direction[1]
            
            # Wrap around screen edges
            if self.x < 0:
                self.x = MAZE_WIDTH - 1
            elif self.x >= MAZE_WIDTH:
                self.x = 0
                
    def can_move(self, maze, direction):
        new_x = self.x + direction[0]
        new_y = self.y + direction[1]
        
        # Check bounds
        if new_x < 0 or new_x >= MAZE_WIDTH or new_y < 0 or new_y >= MAZE_HEIGHT:
            return True  # Allow wrapping
            
        # Check if wall
        return maze[new_y][new_x] != 1
        
    def draw(self, screen):
        # Draw ghost as a colored circle
        center_x = self.x * CELL_SIZE + CELL_SIZE // 2
        center_y = self.y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 2
        
        pygame.draw.circle(screen, self.color, (center_x, center_y), radius)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pacman")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.maze = self.create_maze()
        self.dots = self.create_dots()
        self.pacman = Pacman(1, 1)
        self.ghosts = [
            Ghost(MAZE_WIDTH - 2, 1, RED),
            Ghost(MAZE_WIDTH - 2, MAZE_HEIGHT - 2, PINK),
            Ghost(1, MAZE_HEIGHT - 2, CYAN),
            Ghost(MAZE_WIDTH // 2, MAZE_HEIGHT // 2, ORANGE)
        ]
        
        self.game_state = PLAYING
        
    def create_maze(self):
        # Create a simpler maze layout with more open space
        maze = [[1 for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]
        
        # Create paths with more open space
        for y in range(1, MAZE_HEIGHT - 1):
            for x in range(1, MAZE_WIDTH - 1):
                if random.random() < 0.8:  # 80% chance of being a path (more open)
                    maze[y][x] = 0
                    
        # Ensure starting positions are clear
        maze[1][1] = 0  # Pacman start
        maze[1][MAZE_WIDTH - 2] = 0  # Ghost start
        maze[MAZE_HEIGHT - 2][1] = 0  # Ghost start
        maze[MAZE_HEIGHT - 2][MAZE_WIDTH - 2] = 0  # Ghost start
        maze[MAZE_HEIGHT // 2][MAZE_WIDTH // 2] = 0  # Ghost start
        
        # Clear more space around starting positions
        for y in range(0, 3):
            for x in range(0, 3):
                if 0 <= y < MAZE_HEIGHT and 0 <= x < MAZE_WIDTH:
                    maze[y][x] = 0
                if 0 <= y < MAZE_HEIGHT and 0 <= MAZE_WIDTH - 1 - x < MAZE_WIDTH:
                    maze[y][MAZE_WIDTH - 1 - x] = 0
                if 0 <= MAZE_HEIGHT - 1 - y < MAZE_HEIGHT and 0 <= x < MAZE_WIDTH:
                    maze[MAZE_HEIGHT - 1 - y][x] = 0
                if 0 <= MAZE_HEIGHT - 1 - y < MAZE_HEIGHT and 0 <= MAZE_WIDTH - 1 - x < MAZE_WIDTH:
                    maze[MAZE_HEIGHT - 1 - y][MAZE_WIDTH - 1 - x] = 0
        
        return maze
        
    def create_dots(self):
        dots = []
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                if self.maze[y][x] == 0:  # If it's a path
                    dots.append((x, y))
        return dots
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r and self.game_state != PLAYING:
                    self.__init__()  # Restart game
                    
        # Handle continuous key presses for movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.pacman.set_direction([0, -1])
        elif keys[pygame.K_DOWN]:
            self.pacman.set_direction([0, 1])
        elif keys[pygame.K_LEFT]:
            self.pacman.set_direction([-1, 0])
        elif keys[pygame.K_RIGHT]:
            self.pacman.set_direction([1, 0])
            
        return True
        
    def update(self):
        if self.game_state != PLAYING:
            return
            
        # Move Pacman
        self.pacman.move(self.maze)
        
        # Move ghosts
        for ghost in self.ghosts:
            ghost.move(self.maze)
            
        # Check dot collection
        pacman_pos = (self.pacman.x, self.pacman.y)
        if pacman_pos in self.dots:
            self.dots.remove(pacman_pos)
            self.pacman.score += 10
            
        # Check win condition
        if len(self.dots) == 0:
            self.game_state = WIN
            
        # Check collision with ghosts
        for ghost in self.ghosts:
            if (self.pacman.x, self.pacman.y) == (ghost.x, ghost.y):
                self.pacman.lives -= 1
                if self.pacman.lives <= 0:
                    self.game_state = GAME_OVER
                else:
                    # Reset positions
                    self.pacman.x, self.pacman.y = 1, 1
                    self.ghosts[0].x, self.ghosts[0].y = MAZE_WIDTH - 2, 1
                    self.ghosts[1].x, self.ghosts[1].y = MAZE_WIDTH - 2, MAZE_HEIGHT - 2
                    self.ghosts[2].x, self.ghosts[2].y = 1, MAZE_HEIGHT - 2
                    self.ghosts[3].x, self.ghosts[3].y = MAZE_WIDTH // 2, MAZE_HEIGHT // 2
                break
                
    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw maze
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                if self.maze[y][x] == 1:  # Wall
                    pygame.draw.rect(self.screen, BLUE, 
                                   (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    
        # Draw dots
        for dot in self.dots:
            center_x = dot[0] * CELL_SIZE + CELL_SIZE // 2
            center_y = dot[1] * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, WHITE, (center_x, center_y), 2)
            
        # Draw Pacman
        self.pacman.draw(self.screen)
        
        # Draw ghosts
        for ghost in self.ghosts:
            ghost.draw(self.screen)
            
        # Draw UI
        score_text = self.font.render(f"Score: {self.pacman.score}", True, WHITE)
        lives_text = self.font.render(f"Lives: {self.pacman.lives}", True, WHITE)
        dots_text = self.font.render(f"Dots: {len(self.dots)}", True, WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 50))
        self.screen.blit(dots_text, (10, 90))
        
        # Draw game state messages
        if self.game_state == GAME_OVER:
            game_over_text = self.font.render("GAME OVER! Press R to restart", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
        elif self.game_state == WIN:
            win_text = self.font.render("YOU WIN! Press R to restart", True, YELLOW)
            text_rect = win_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(win_text, text_rect)
            
        pygame.display.flip()
        
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)  # 60 FPS
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
