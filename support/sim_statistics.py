import pandas as pd
import os
import argparse
from scipy.stats import hmean
import re
import csv

def determine_category(file_name):
    """
    Determines the category based on the file name.

    Parameters:
        file_name (str): File name.

    Returns:
        str: Identified category.
    """
    if "sz10_s10_py10_pd5_o0.1" in file_name:
        return "standard"
    elif "sz20_s10_py15_pd10_o0.05" in file_name:
        return "extensive_sparse"
    elif "sz10_s10_py15_pd10_o0.3" in file_name:
        return "dense_constrained"
    else:
        return ""

def determine_algorithm(file_name):
    """
    Determines the algorithm based on the file name.

    Parameters:
        file_name (str): File name.

    Returns:
        str: Algoritmo identificado.
    """
    if "dueling" in file_name:
        return "dueling"
    elif "double" in file_name:
        return "double"
    else:
        return "dqn"

def determine_type(file_name):
    """
    Determina o tipo com base no nome do arquivo.

    Parameters:
        file_name (str): File name.

    Returns:
        str: Tipo identificado.
    """
    if "radar" in file_name:
        return "radar"
    else:
        return "tradicional"

def load_and_split_data_old(file_path):
    """
    Carrega os dados do arquivo e separa em DataFrames para preys e predatores.

    Parameters:
        file_path (str): Caminho para o arquivo de dados.

    Returns:
        tuple: Dois DataFrames, um para preys e outro para predatores.
    """
    import csv
    import pandas as pd

    # Initialize lists to store the data
    data = {
        'episode': [],
        'step': [],
        'breed': [],
        'feedback': [],
        'reward': [],
        'done': [],
        'coordinate': [],  # To store the extracted coordinates
    }

    # Read the file line by line
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            log = row[0]  # Each row contains the full string

            # Extract the fields manually
            try:
                episode = log.split("/")[0]
                step = int(log.split(":")[0].split("/")[1])
                breed = int(log.split("Breed:")[1].split(",")[0])
                feedback = log.split("Feedback: ")[1] if "Feedback: " in log else None
                reward = float(log.split("Reward: ")[1].split(",")[0])
                done = log.split("Done: ")[1].split(",")[0].strip().lower() == 'true'

                # Try to extract coordinates from the log (pattern like "(x, y)")
                coordinate = None
                if "Position: " in log:
                    match = re.search(r'Position: \((\d+,\d+)\)', log)
                    if match:
                        coordinate = match.group(1)

                # Add the extracted fields to the dictionary
                data['episode'].append(episode)
                data['step'].append(step)
                data['breed'].append(breed)
                data['feedback'].append(feedback)
                data['reward'].append(reward)
                data['done'].append(done)
                data['coordinate'].append(coordinate)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Erro ao processar a linha: {log}\nErro: {e}")

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)

    # Split into prey and predators
    prey_data = df[df['breed'] == 2].copy()
    predator_data = df[df['breed'] == 0].copy()

    return prey_data, predator_data

def load_and_split_data(file_path):
    """
    Carrega os dados do arquivo e separa em DataFrames para preys e predatores.

    Parameters:
        file_path (str): Caminho para o arquivo de dados.

    Returns:
        tuple: Dois DataFrames, um para preys e outro para predatores.
    """
    # Initialize lists to store the data
    data = {
        'episode': [],
        'step': [],
        'breed': [],
        'feedback': [],
        'reward': [],
        'done': [],
        'coordinate': [],  # To store the extracted coordinates
    }

    # Read the file line by line
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            log = row[0]  # Each row contains the full string

            # Extract the fields manually
            try:
                episode = log.split("/")[0]
                step = int(log.split(":")[0].split("/")[1])
                breed_match = re.search(r'Breed:(\d+)', log)
                breed = int(breed_match.group(1)) if breed_match else None
                feedback = log.split("Feedback: ")[1] if "Feedback: " in log else None
                reward_match = re.search(r'Reward: ([\d\.\-]+)', log)
                reward = float(reward_match.group(1)) if reward_match else None
                done_match = re.search(r'Done: (True|False)', log)
                done = done_match.group(1).lower() == 'true' if done_match else None

                # Try to extract coordinates from the log (pattern like "(x, y)")
                coordinate = None
                match = re.search(r'Position: \((\d+), (\d+)\)', log)
                if match:
                    coordinate = f"({match.group(1)}, {match.group(2)})"

                # Add the extracted fields to the dictionary
                data['episode'].append(episode)
                data['step'].append(step)
                data['breed'].append(breed)
                data['feedback'].append(feedback)
                data['reward'].append(reward)
                data['done'].append(done)
                data['coordinate'].append(coordinate)

            except (IndexError, ValueError, AttributeError) as e:
                raise ValueError(f"Erro ao processar a linha: {log}\nErro: {e}")

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)

    # Split into prey and predators
    prey_data = df[df['breed'] == 2].copy()
    predator_data = df[df['breed'] == 0].copy()

    return prey_data, predator_data

def calcular_total_sucesso_escapes(df):
    """
    Calcula o total de feedbacks de sucesso para "[PREY]: Evasao do alvo" no DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados das preys.

    Returns:
        int: Total de feedbacks de sucesso contendo "[PREY]: Evasao do alvo".
    """
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    total_escapes = filtro.sum()
    return total_escapes

def calcular_total_sucesso_capturas(df):
    """
    Calcula o total de feedbacks de sucesso para "[PREDATOR]: Alvo capturado" no DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados dos predatores.

    Returns:
        int: Total de feedbacks de sucesso contendo "[PREDATOR]: Alvo capturado".
    """
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    total_capturas = filtro.sum()
    return total_capturas

def calcular_media_sucesso_escapes(df):
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    df_filtrado = df[filtro]

    # Group by episode and compute the average successes per step
    #average_per_episode = df_filtered.groupby('episode').size() / df.groupby('episode')['step'].max()
    media_per_episode = df_filtrado.groupby('episode').size()

    # Compute the overall average across episodes
    media_geral = media_per_episode.mean()
    return round(media_geral, 3)

def calcular_media_sucesso_capturas(df):
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    df_filtrado = df[filtro]

    # Group by episode and compute the average successes per step
    #average_per_episode = df_filtered.groupby('episode').size() / df.groupby('episode')['step'].max()
    media_per_episode = df_filtrado.groupby('episode').size()

    # Compute the overall average across episodes
    media_geral = media_per_episode.mean()
    return round(media_geral, 3)

def calcular_mediana_sucesso_escapes(df):
    """
    Computes the median of successes per step for "[PREY]: Target evasion" in each episode and,
    then computes the median across episodes.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados das preys.

    Returns:
        float: Mediana de sucessos por etapa para "[PREY]: Evasao do alvo".
    """
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    df_filtrado = df[filtro]

    # Group by episode and compute the proportion of successes per step
    proporcao_per_episode = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Compute the median across episodes
    mediana_geral = proporcao_per_episode.median()
    return round(mediana_geral, 3)

def calcular_mediana_sucesso_capturas(df):
    """
    Computes the median of successes per step for "[PREDATOR]: Target captured" in each episode and,
    then computes the median across episodes.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados dos predatores.

    Returns:
        float: Mediana de sucessos por etapa para "[PREDATOR]: Alvo capturado".
    """
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    df_filtrado = df[filtro]

    # Group by episode and compute the proportion of successes per step
    proporcao_per_episode = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Compute the median across episodes
    mediana_geral = proporcao_per_episode.median()
    return round(mediana_geral, 3)

def compute_std_success_escapes(df):
    """
    Computes the standard deviation of successes per step for "[PREY]: Target evasion" in each episode and,
    then computes the standard deviation across episodes.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados das preys.

    Returns:
        float: Standard deviation of successes per step for "[PREY]: Target evasion".
    """
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    df_filtrado = df[filtro]

    # Group by episode and compute the proportion of successes per step
    proporcao_per_episode = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Compute the standard deviation across episodes
    overall_std = proporcao_per_episode.std()
    return round(overall_std, 3)

def compute_std_success_captures(df):
    """
    Computes the standard deviation of successes per step for "[PREDATOR]: Target captured" in each episode and,
    then computes the standard deviation across episodes.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados dos predatores.

    Returns:
        float: Standard deviation of successes per step for "[PREDATOR]: Target captured".
    """
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    df_filtrado = df[filtro]

    # Group by episode and compute the proportion of successes per step
    proporcao_per_episode = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Compute the standard deviation across episodes
    overall_std = proporcao_per_episode.std()
    return round(overall_std, 3)

def calcular_media_harmonica_sucesso_escapes(df):
    """
    Computes the harmonic mean of successes per step for "[PREY]: Target evasion" in each episode and,
    then computes the harmonic mean across episodes.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados das preys.

    Returns:
        float: Harmonic mean of successes per step for "[PREY]: Target evasion".
    """
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    df_filtrado = df[filtro]

    # Group by episode and compute the proportion of successes per step
    proporcao_per_episode = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Filter positive values (harmonic mean only accepts positive values)
    proporcao_valida = proporcao_per_episode[proporcao_per_episode > 0]

    # Compute the harmonic mean across episodes
    if not proporcao_valida.empty:
        media_harmonica_geral = hmean(proporcao_valida)
    else:
        media_harmonica_geral = 0

    return round(media_harmonica_geral, 3)

def calcular_media_harmonica_sucesso_capturas(df):
    """
    Computes the harmonic mean of successes per step for "[PREDATOR]: Target captured" in each episode and,
    then computes the harmonic mean across episodes.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados dos predatores.

    Returns:
        float: Harmonic mean of successes per step for "[PREDATOR]: Target captured".
    """
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    df_filtrado = df[filtro]

    # Group by episode and compute the proportion of successes per step
    proporcao_per_episode = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Filter positive values (harmonic mean only accepts positive values)
    proporcao_valida = proporcao_per_episode[proporcao_per_episode > 0]

    # Compute the harmonic mean across episodes
    if not proporcao_valida.empty:
        media_harmonica_geral = hmean(proporcao_valida)
    else:
        media_harmonica_geral = 0

    return round(media_harmonica_geral, 3)

def calcular_media_proximidade_aliados_preys(df):
    """
    Computes the average ally proximity for prey per step, considering different proximity levels.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados das preys.

    Returns:
        float: Overall average ally proximity for prey.
    """
    medias_por_proximidade = [
        df[df['feedback'].str.contains(rf'\[PREY\]: Proximidade aliado: {i}', na=False)]
        .groupby(['episode', 'step'])
        .size()
        .groupby('episode')
        .mean()
        .mean()
        for i in range(1, 4)
    ]

    # Compute the overall average of the proximities
    media_geral = sum(medias_por_proximidade) / len(medias_por_proximidade)
    return round(media_geral, 3)

def calcular_media_proximidade_aliados_predatores(df):
    """
    Computes the average ally proximity for predators per step, considering different proximity levels.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados dos predatores.

    Returns:
        float: Overall average ally proximity for predators.
    """
    medias_por_proximidade = [
        df[df['feedback'].str.contains(rf'\[PREDATOR\]: Proximidade aliado: {i}', na=False)]
        .groupby(['episode', 'step'])
        .size()
        .groupby('episode')
        .mean()
        .mean()
        for i in range(1, 4)
    ]

    # Compute the overall average of the proximities
    media_geral = sum(medias_por_proximidade) / len(medias_por_proximidade)
    return round(media_geral, 3)

def calcular_media_coordinates_unicas(df):
    """
    Computes the average number of unique coordinates visited per episode.

    Parameters:
        df (pd.DataFrame): DataFrame com os dados.

    Returns:
        float: Average number of unique coordinates visited per episode.
    """
    if df.empty:
        return 0

    # Validate that the 'coordinate' column was filled in
    if 'coordinate' not in df.columns or df['coordinate'].isnull().all():
        raise ValueError("The 'coordinate' column is empty or was not filled in correctly.")

    # Get unique coordinates per episode
    unique_coordinates_per_episode = df.groupby('episode')['coordinate'].apply(lambda x: len(set(x.dropna())))

    # Compute the average of unique coordinates per episode
    media_coordinates = unique_coordinates_per_episode.mean()
    return round(media_coordinates, 3)


def count_total_coordinates(df, coordinates_column="coordinate", per_episode=True):
    """
    Conta o total de coordinates visitadas.

    Parameters:
        df (pd.DataFrame): DataFrame containing the simulation data.
        coordinates_column (str): Name of the column where the coordinates are stored.
        per_episode (bool): If True, returns the total unique coordinates per episode.

    Returns:
        int: Total de coordinates visitadas.
    """
    if coordinates_column not in df.columns:
        raise ValueError(f"A coluna '{coordinates_column}' is not in the DataFrame.")

    # Make sure the coordinates column has no null values
    df = df.dropna(subset=[coordinates_column])

    if per_episode:
        coordinates_per_episode = (
            df.groupby('episode')[coordinates_column]
            .apply(lambda x: len(x))  # Total visited coordinates
            .sum()
        )
        return coordinates_per_episode
    else:
        total_coordinates = len(df[coordinates_column])  # Sum of all visited coordinates
        return total_coordinates



def process_files(file_paths):
    """
    Processes a list of files and computes the relevant statistics.

    Parameters:
        file_paths (list): Lista de caminhos para os arquivos.

    Returns:
        pd.DataFrame: DataFrame with the computed statistics.
    """
    summary_rows = []

    for file_path in file_paths:
        prey_data, predator_data = load_and_split_data(file_path)

        # total success by breed function and feedback
        prey_total = calcular_total_sucesso_escapes(prey_data)
        predator_total = calcular_total_sucesso_capturas(predator_data)

        
        # average success per done
        prey_mean_done = calcular_media_sucesso_escapes(prey_data)
        predator_mean_done = calcular_media_sucesso_capturas(predator_data) 

        # harmonic average of success per done
        prey_harm_done = calcular_media_harmonica_sucesso_escapes(prey_data)
        predator_harm_done = calcular_media_harmonica_sucesso_capturas(predator_data) 

        # median of success per done
        prey_mediana_done = calcular_mediana_sucesso_escapes(prey_data)
        predator_mediana_done = calcular_mediana_sucesso_capturas(predator_data)

        # standard deviation of success per done
        prey_dvp_done = compute_std_success_escapes(prey_data)
        predator_dvp_done = compute_std_success_captures(predator_data)

        # nearby ally average
        prey_media_proximidade_aliados = calcular_media_proximidade_aliados_preys(prey_data)
        predator_media_proximidade_aliados = calcular_media_proximidade_aliados_predatores(predator_data)

        # average unique coordinates
        prey_media_coord_unicas = calcular_media_coordinates_unicas(prey_data)
        predatores_media_coord_unicas = calcular_media_coordinates_unicas(predator_data)

        prey_media_coord = count_total_coordinates(prey_data)
        predatores_media_coord = count_total_coordinates(predator_data)

        summary_row = {
            'file_name': os.path.basename(file_path),
            'category': determine_category(os.path.basename(file_path)),
            'algorithm': determine_algorithm(os.path.basename(file_path)),
            'type': determine_type(os.path.basename(file_path)),

            '[PREY]: total de done': prey_total,
            '[PREDATOR]: total de done': predator_total,
            '[SOMA]: total de done': predator_total + prey_total,
            '[PREY]: media escapes done': prey_mean_done,
            '[PREDATOR]: media captures done': predator_mean_done,
            '[SOMA]: media de done': predator_mean_done + prey_mean_done,
            '[PREY]: harmonica escapes done': prey_harm_done,
            '[PREDATOR]: harmonica captures done': predator_harm_done,
            '[PREY]: mediana escapes done': prey_mediana_done,
            '[PREDATOR]: mediana captures done': predator_mediana_done,
            '[PREY]: dvp escapes done': prey_dvp_done,
            '[PREDATOR]: dvp captures done': predator_dvp_done,
            '[PREY]: media proximidade aliados': prey_media_proximidade_aliados,
            '[PREDATOR]: media proximidade aliados': predator_media_proximidade_aliados,
            '[SOMA]: media proximidade aliados': prey_media_proximidade_aliados + predator_media_proximidade_aliados,
            '[PREY]: media efetividade aliados': round(prey_mean_done / prey_media_proximidade_aliados, 3),
            '[PREDATOR]: media efetividade aliados': round(predator_mean_done / predator_media_proximidade_aliados, 3),
            '[SOMA]: media efetividade aliados': round(predator_mean_done / predator_media_proximidade_aliados, 3) + round(prey_mean_done / prey_media_proximidade_aliados, 3),
            '[PREY]: media coordinates unicas': prey_media_coord_unicas,
            '[PREDATOR]: media coordinates unicas': predatores_media_coord_unicas,
            '[SOMA]: media coordinates unicas': predatores_media_coord_unicas + prey_media_coord_unicas,
            '[PREY]: media coordinates': prey_media_coord,
            '[PREDATOR]: media coordinates': predatores_media_coord,
            '[SOMA]: media coordinates': predatores_media_coord + prey_media_coord,

        }


        summary_rows.append(summary_row)

    df_summary = pd.DataFrame(summary_rows)

    # Replace dots with commas in numeric values
    for col in df_summary.select_dtypes(include=['float']).columns:
        df_summary[col] = df_summary[col].apply(lambda x: str(x).replace('.', ','))

    return df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process simulation files and generate statistics.")
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True,
        help="Directory containing the simulation files."
    )
    parser.add_argument(
        "-o", "--output_file", type=str, required=True,
        help="Path to the output CSV file."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    file_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith('sim')
    ]

    if not file_paths:
        print("No file found in the specified directory.")
        exit(1)

    summary_df = process_files(file_paths)

    if not summary_df.empty:
        summary_df.to_csv(args.output_file, index=False, sep=';')
        print(f"Resumo salvo em: {args.output_file}")
    else:
        print("Nenhum dado processado. Verifique os arquivos de entrada.")
