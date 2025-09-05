"""
Heatmap Generation Module
Creates visual heatmaps and analysis plots for football tracking data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import ndimage
from scipy.stats import gaussian_kde
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import cv2

class FootballHeatmapGenerator:
    """
    Generates heatmaps and visualizations for football tracking data.
    """
    
    def __init__(self, pitch_length: float = 100.0, pitch_width: float = 64.0, 
                 dpi: int = 100):
        """
        Initialize heatmap generator.
        
        Args:
            pitch_length: Length of pitch in meters
            pitch_width: Width of pitch in meters
            dpi: DPI for output images
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.dpi = dpi
        
        # Set up custom colormap for heatmaps
        self.heatmap_cmap = self._create_heatmap_colormap()
        
        # Pitch colors
        self.pitch_color = '#2d5a2d'  # Dark green
        self.line_color = 'white'
        self.line_width = 2
        
    def _create_heatmap_colormap(self) -> LinearSegmentedColormap:
        """
        Create custom colormap for football heatmaps.
        
        Returns:
            Custom colormap
        """
        colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('football_heat', colors, N=n_bins)
        return cmap
    
    def create_pitch_figure(self, figsize: Tuple[float, float] = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a figure with football pitch background.
        
        Args:
            figsize: Figure size in inches
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Set pitch background
        ax.set_facecolor(self.pitch_color)
        
        # Draw pitch markings
        self._draw_pitch_markings(ax)
        
        # Set equal aspect ratio and limits
        ax.set_xlim(0, self.pitch_length)
        ax.set_ylim(0, self.pitch_width)
        ax.set_aspect('equal')
        
        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig, ax
    
    def _draw_pitch_markings(self, ax: plt.Axes):
        """
        Draw standard football pitch markings.
        
        Args:
            ax: Matplotlib axes to draw on
        """
        # Pitch boundary
        boundary = patches.Rectangle((0, 0), self.pitch_length, self.pitch_width,
                                   linewidth=self.line_width, edgecolor=self.line_color,
                                   facecolor='none')
        ax.add_patch(boundary)
        
        # Center line
        ax.plot([self.pitch_length/2, self.pitch_length/2], [0, self.pitch_width],
                color=self.line_color, linewidth=self.line_width)
        
        # Center circle
        center_circle = patches.Circle((self.pitch_length/2, self.pitch_width/2), 9.15,
                                     linewidth=self.line_width, edgecolor=self.line_color,
                                     facecolor='none')
        ax.add_patch(center_circle)
        
        # Center spot
        ax.plot(self.pitch_length/2, self.pitch_width/2, 'o', color=self.line_color, markersize=3)
        
        # Penalty areas
        # Left penalty area
        left_penalty = patches.Rectangle((0, (self.pitch_width-40.32)/2), 16.5, 40.32,
                                       linewidth=self.line_width, edgecolor=self.line_color,
                                       facecolor='none')
        ax.add_patch(left_penalty)
        
        # Right penalty area
        right_penalty = patches.Rectangle((self.pitch_length-16.5, (self.pitch_width-40.32)/2), 
                                        16.5, 40.32,
                                        linewidth=self.line_width, edgecolor=self.line_color,
                                        facecolor='none')
        ax.add_patch(right_penalty)
        
        # Goal areas (6-yard boxes)
        # Left goal area
        left_goal = patches.Rectangle((0, (self.pitch_width-18.32)/2), 5.5, 18.32,
                                    linewidth=self.line_width, edgecolor=self.line_color,
                                    facecolor='none')
        ax.add_patch(left_goal)
        
        # Right goal area
        right_goal = patches.Rectangle((self.pitch_length-5.5, (self.pitch_width-18.32)/2), 
                                     5.5, 18.32,
                                     linewidth=self.line_width, edgecolor=self.line_color,
                                     facecolor='none')
        ax.add_patch(right_goal)
        
        # Penalty spots
        ax.plot(11, self.pitch_width/2, 'o', color=self.line_color, markersize=3)
        ax.plot(self.pitch_length-11, self.pitch_width/2, 'o', color=self.line_color, markersize=3)
        
        # Corner arcs
        corner_radius = 1
        # Bottom-left
        corner1 = patches.Wedge((0, 0), corner_radius, 0, 90,
                              linewidth=self.line_width, edgecolor=self.line_color,
                              facecolor='none')
        ax.add_patch(corner1)
        
        # Bottom-right
        corner2 = patches.Wedge((self.pitch_length, 0), corner_radius, 90, 180,
                              linewidth=self.line_width, edgecolor=self.line_color,
                              facecolor='none')
        ax.add_patch(corner2)
        
        # Top-right
        corner3 = patches.Wedge((self.pitch_length, self.pitch_width), corner_radius, 180, 270,
                              linewidth=self.line_width, edgecolor=self.line_color,
                              facecolor='none')
        ax.add_patch(corner3)
        
        # Top-left
        corner4 = patches.Wedge((0, self.pitch_width), corner_radius, 270, 360,
                              linewidth=self.line_width, edgecolor=self.line_color,
                              facecolor='none')
        ax.add_patch(corner4)
    
    def generate_player_heatmap(self, positions: List[List[float]], player_id: int,
                              output_path: Optional[str] = None, 
                              title: Optional[str] = None,
                              sigma: float = 2.0) -> plt.Figure:
        """
        Generate heatmap for a single player.
        
        Args:
            positions: List of [x, y] positions in pitch coordinates
            player_id: Player ID for labeling
            output_path: Optional path to save image
            title: Optional custom title
            sigma: Gaussian blur sigma for smoothing
            
        Returns:
            Matplotlib figure
        """
        if not positions:
            raise ValueError("No positions provided")
        
        fig, ax = self.create_pitch_figure()
        
        # Convert positions to numpy array
        pos_array = np.array(positions)
        
        # Create 2D histogram
        x_bins = np.linspace(0, self.pitch_length, 100)
        y_bins = np.linspace(0, self.pitch_width, 64)
        
        heatmap, _, _ = np.histogram2d(pos_array[:, 0], pos_array[:, 1],
                                     bins=[x_bins, y_bins])
        
        # Apply Gaussian smoothing
        heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Plot heatmap
        extent = [0, self.pitch_length, 0, self.pitch_width]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                      cmap=self.heatmap_cmap, alpha=0.7, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Activity Density', rotation=270, labelpad=15)
        
        # Set title
        if title is None:
            title = f'Player {player_id} Heatmap ({len(positions)} positions)'
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', facecolor='black')
            print(f"Player heatmap saved to {output_path}")
        
        return fig
    
    def generate_team_heatmap(self, team_positions: Dict[int, List[List[float]]],
                            team_name: str = "Team",
                            output_path: Optional[str] = None,
                            sigma: float = 2.0) -> plt.Figure:
        """
        Generate combined heatmap for a team.
        
        Args:
            team_positions: Dictionary mapping player_id to positions
            team_name: Team name for labeling
            output_path: Optional path to save image
            sigma: Gaussian blur sigma for smoothing
            
        Returns:
            Matplotlib figure
        """
        if not team_positions:
            raise ValueError("No team positions provided")
        
        fig, ax = self.create_pitch_figure()
        
        # Combine all positions
        all_positions = []
        total_players = 0
        for player_id, positions in team_positions.items():
            all_positions.extend(positions)
            total_players += 1
        
        if not all_positions:
            raise ValueError("No positions in team data")
        
        # Convert to numpy array
        pos_array = np.array(all_positions)
        
        # Create 2D histogram
        x_bins = np.linspace(0, self.pitch_length, 100)
        y_bins = np.linspace(0, self.pitch_width, 64)
        
        heatmap, _, _ = np.histogram2d(pos_array[:, 0], pos_array[:, 1],
                                     bins=[x_bins, y_bins])
        
        # Apply Gaussian smoothing
        heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Plot heatmap
        extent = [0, self.pitch_length, 0, self.pitch_width]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower',
                      cmap=self.heatmap_cmap, alpha=0.7, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Team Activity Density', rotation=270, labelpad=15)
        
        # Set title
        title = f'{team_name} Heatmap ({total_players} players, {len(all_positions)} positions)'
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', facecolor='black')
            print(f"Team heatmap saved to {output_path}")
        
        return fig
    
    def generate_comparison_heatmap(self, team1_positions: Dict[int, List[List[float]]],
                                  team2_positions: Dict[int, List[List[float]]],
                                  team1_name: str = "Team 1",
                                  team2_name: str = "Team 2",
                                  output_path: Optional[str] = None,
                                  sigma: float = 2.0) -> plt.Figure:
        """
        Generate side-by-side comparison heatmap for two teams.
        
        Args:
            team1_positions: Team 1 positions
            team2_positions: Team 2 positions
            team1_name: Team 1 name
            team2_name: Team 2 name
            output_path: Optional path to save image
            sigma: Gaussian blur sigma
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=self.dpi)
        
        # Team 1 heatmap
        self._draw_pitch_markings(ax1)
        ax1.set_facecolor(self.pitch_color)
        ax1.set_xlim(0, self.pitch_length)
        ax1.set_ylim(0, self.pitch_width)
        ax1.set_aspect('equal')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Team 2 heatmap
        self._draw_pitch_markings(ax2)
        ax2.set_facecolor(self.pitch_color)
        ax2.set_xlim(0, self.pitch_length)
        ax2.set_ylim(0, self.pitch_width)
        ax2.set_aspect('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Process team 1
        if team1_positions:
            all_pos1 = []
            for positions in team1_positions.values():
                all_pos1.extend(positions)
            
            if all_pos1:
                pos_array1 = np.array(all_pos1)
                x_bins = np.linspace(0, self.pitch_length, 100)
                y_bins = np.linspace(0, self.pitch_width, 64)
                heatmap1, _, _ = np.histogram2d(pos_array1[:, 0], pos_array1[:, 1],
                                             bins=[x_bins, y_bins])
                heatmap1 = ndimage.gaussian_filter(heatmap1, sigma=sigma)
                if heatmap1.max() > 0:
                    heatmap1 = heatmap1 / heatmap1.max()
                
                extent = [0, self.pitch_length, 0, self.pitch_width]
                im1 = ax1.imshow(heatmap1.T, extent=extent, origin='lower',
                               cmap=self.heatmap_cmap, alpha=0.7, aspect='auto')
        
        # Process team 2
        if team2_positions:
            all_pos2 = []
            for positions in team2_positions.values():
                all_pos2.extend(positions)
            
            if all_pos2:
                pos_array2 = np.array(all_pos2)
                heatmap2, _, _ = np.histogram2d(pos_array2[:, 0], pos_array2[:, 1],
                                             bins=[x_bins, y_bins])
                heatmap2 = ndimage.gaussian_filter(heatmap2, sigma=sigma)
                if heatmap2.max() > 0:
                    heatmap2 = heatmap2 / heatmap2.max()
                
                im2 = ax2.imshow(heatmap2.T, extent=extent, origin='lower',
                               cmap=self.heatmap_cmap, alpha=0.7, aspect='auto')
        
        # Set titles
        ax1.set_title(f'{team1_name} Heatmap', fontsize=14, fontweight='bold', 
                     color='white', pad=20)
        ax2.set_title(f'{team2_name} Heatmap', fontsize=14, fontweight='bold',
                     color='white', pad=20)
        
        # Add shared colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label('Activity Density', rotation=270, labelpad=15)
        
        # Set figure background
        fig.patch.set_facecolor('black')
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', facecolor='black')
            print(f"Comparison heatmap saved to {output_path}")
        
        return fig
    
    def generate_movement_paths(self, player_positions: Dict[int, List[List[float]]],
                              output_path: Optional[str] = None,
                              max_players: int = 5) -> plt.Figure:
        """
        Generate movement path visualization for players.
        
        Args:
            player_positions: Dictionary mapping player_id to positions
            output_path: Optional path to save image
            max_players: Maximum number of players to show
            
        Returns:
            Matplotlib figure
        """
        fig, ax = self.create_pitch_figure()
        
        # Color palette for different players
        colors = plt.cm.Set3(np.linspace(0, 1, max_players))
        
        player_count = 0
        for player_id, positions in player_positions.items():
            if player_count >= max_players:
                break
            
            if len(positions) < 2:
                continue
            
            pos_array = np.array(positions)
            color = colors[player_count]
            
            # Plot path
            ax.plot(pos_array[:, 0], pos_array[:, 1], 
                   color=color, alpha=0.7, linewidth=2, 
                   label=f'Player {player_id}')
            
            # Mark start and end points
            ax.scatter(pos_array[0, 0], pos_array[0, 1], 
                      color=color, s=100, marker='o', edgecolor='white', linewidth=2)
            ax.scatter(pos_array[-1, 0], pos_array[-1, 1], 
                      color=color, s=100, marker='s', edgecolor='white', linewidth=2)
            
            player_count += 1
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), facecolor='white')
        
        # Set title
        ax.set_title('Player Movement Paths', fontsize=14, fontweight='bold', 
                    color='white', pad=20)
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', facecolor='black')
            print(f"Movement paths saved to {output_path}")
        
        return fig
    
    def save_pitch_template(self, output_path: str, figsize: Tuple[float, float] = (12, 8)):
        """
        Save a clean pitch template image.
        
        Args:
            output_path: Path to save template
            figsize: Figure size
        """
        fig, ax = self.create_pitch_figure(figsize)
        ax.set_title('Football Pitch Template', fontsize=16, fontweight='bold',
                    color='white', pad=20)
        
        plt.savefig(output_path, bbox_inches='tight', facecolor='black', dpi=self.dpi)
        plt.close(fig)
        print(f"Pitch template saved to {output_path}")

# Test function
def test_heatmap_generator():
    """Test the heatmap generator with sample data."""
    generator = FootballHeatmapGenerator()
    
    # Generate sample positions
    np.random.seed(42)
    sample_positions = []
    for _ in range(100):
        x = np.random.normal(50, 15)  # Centered around midfield
        y = np.random.normal(32, 10)  # Centered vertically
        x = np.clip(x, 0, 100)
        y = np.clip(y, 0, 64)
        sample_positions.append([x, y])
    
    # Test player heatmap
    fig = generator.generate_player_heatmap(sample_positions, player_id=1)
    print("Generated sample player heatmap")
    
    # Test pitch template
    generator.save_pitch_template("test_pitch_template.png")
    
    plt.close('all')
    print("HeatmapGenerator test completed!")

if __name__ == "__main__":
    test_heatmap_generator()