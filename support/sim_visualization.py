import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import argparse

def visualizar_simulacao(csv_filename, speed=100):
    # Relative path to the CSV file
    csv_path = os.path.abspath(csv_filename)

    # Check whether the file exists before loading
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

    # Load the data
    df = pd.read_csv(csv_path)

    # Create the 'Channel' column by combining R, G, and B
    df['Channel'] = df.apply(lambda row: [row['R'] / 255, row['G'] / 255, row['B'] / 255], axis=1)

    # Generate 'Frame' based on 'Episode' and 'Step'
    df['Frame'] = df.groupby(['Episode', 'Step']).ngroup()

    # Determine the grid size
    grid_width = df['Grid_Width'].iloc[0]
    grid_height = df['Grid_Height'].iloc[0]

    # Validate that all objects are within the grid
    if df['X'].max() > grid_width or df['Y'].max() > grid_height:
        raise ValueError("Some objects are outside the grid boundaries.")

    # Separate obstacles (fixed) from agents (dynamic)
    obstacles = df[df['Type'] == 'Obstacle']
    agents = df[df['Type'] != 'Obstacle']

    # Initial animation configuration
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('dimgray')  # Figure background
    ax.set_facecolor('black')  # Chart background

    # Configure the chart limits based on the grid
    ax.set_xlim(-1, grid_width)
    ax.set_ylim(-1, grid_height)
    ax.margins(0)
    ax.axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
    index_text_top = fig.text(0.5, 0.95, '', color='white', ha='center', va='top')

    # Scatter plot for obstacles (fixed)
    scatter_obstacles = ax.scatter(obstacles['X'], obstacles['Y'], c=obstacles['Channel'].tolist(), s=30)

    # Initialize scatter plots for dynamic agents
    scatter_agents = {}
    for agent_type in agents['Type'].unique():
        scatter_agents[agent_type] = ax.scatter([], [], s=50)

    def init():
        # Initialize the texts and the charts
        index_text_top.set_text('')
        for scatter in scatter_agents.values():
            scatter.set_offsets(np.empty((0, 2)))
        return [scatter_obstacles] + list(scatter_agents.values()) + [index_text_top]

    def update(frame):
        # Current frame data
        current_df = df[df['Frame'] == frame]
        episode = current_df['Episode'].iloc[0]
        step = current_df['Step'].iloc[0]

        # Update the top index text
        index_text_top.set_text(f'Episode: {episode}, Step: {step}')

        # Update dynamic agents
        for agent_type, scatter in scatter_agents.items():
            # Only living agents (Is_Alive=True) are displayed
            group = current_df[(current_df['Type'] == agent_type) & (current_df['Is_Alive'] == True)]
            scatter_coords = group[['X', 'Y']].values if not group.empty else np.empty((0, 2))
            scatter_colors = group['Channel'].tolist() if not group.empty else []
            scatter.set_offsets(scatter_coords)
            scatter.set_color(scatter_colors)
            scatter.set_visible(True)

        return [scatter_obstacles] + list(scatter_agents.values()) + [index_text_top]

    # Create animation
    ani = FuncAnimation(fig, update, frames=df['Frame'].unique(), init_func=init, interval=speed)

    plt.show()



def preparar_e_chamar_visualizacao(nome_do_arquivo_csv, speed=100):
    if not nome_do_arquivo_csv:
        print("Erro: Nenhum nome de arquivo CSV fornecido.")
        return

    # Call the visualization function directly
    visualizar_simulacao(nome_do_arquivo_csv, speed=speed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize the agent simulation from a CSV file.')
    parser.add_argument('--speed', type=int, default=100, help='Intervalo entre frames em milissegundos (maior = mais lento)')
    parser.add_argument('csv_filename', type=str, help='Path to the CSV file to be used in the simulation.')
    args = parser.parse_args()

    preparar_e_chamar_visualizacao(args.csv_filename, speed=args.speed)
