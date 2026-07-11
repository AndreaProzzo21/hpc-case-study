"""
Caso di Studio HPC: Simulazione del Ciclo di Vita di Pompe Industriali.
Librerie: mpi4py (MPI wrap per Python) e NumPy (vettorizzazione buffer).
"""

import os
import numpy as np
from mpi4py import MPI

FILE_CONFIG = "config.txt"
FILE_OUTPUT = "pumps_results.npy"

def load_or_create_config(filename):
    """
    Gestione I/O centralizzata. Letta solo dal processo Root per evitare 
    concorrenza sul file system e colli di bottiglia hardware.
    """
    defaults = {
        "num_pumps": 1000000,
        "total_cycles": 1000,
        "total_life": 10000.0,
        "base_v_rms": 1.6  # Vibrazione di base
    }
    
    if not os.path.exists(filename):
        print(f"File di configurazione '{filename}' non trovato. Generazione default...")
        with open(filename, "w") as f:
            for key, value in defaults.items():
                f.write(f"{key}={value}\n")
        return defaults
    
    config = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=")
                if "." in val:
                    config[key.strip()] = float(val.strip())
                else:
                    config[key.strip()] = int(val.strip())
    return config

def calculate_partitions(total_elements, size):
    """
    Calcola il Load Balancing per domini non perfettamente divisibili.
    Restituisce i vettori di Count e Displacement necessari per Scatterv e Gatherv.
    """
    counts = np.full(size, total_elements // size, dtype=int)
    # Distribuisce il resto equamente tra i primi processi
    counts[:total_elements % size] += 1
    
    # Calcola l'offset di memoria per ogni blocco (Displacements)
    displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]
    return counts, displacements

def local_simulation(local_health, local_cycles, rank, local_n, config):
    """
    Motore Fisico. Viene eseguito in isolamento da ogni MPI rank.
    """

    np.random.seed(42 + rank)
    
    total_cycles = int(config["total_cycles"])
    total_life = config["total_life"]
    base_v_rms = config["base_v_rms"]
    
    local_temp = np.zeros(local_n, dtype=np.float64)

    # Ciclo temporale (Time-stepping)
    for _ in range(total_cycles):
        local_cycles += 1.0
        
        # 1. Calcolo usura (Curva simil-Weibull vettorizzata su NumPy)
        life_consumed = np.clip(local_cycles / total_life, 0.0, 1.0)
        local_health = np.maximum(0.0, 100.0 * (1.0 - np.power(life_consumed, 2.5)))
        
        # 2. Propagazione fisica (Dall'usura alla vibrazione, al calore)
        wear_f = (100.0 - local_health) / 100.0
        wear_vib = np.power(wear_f, 2.0) * 10.0
        v_rms = base_v_rms + wear_vib
        
        local_temp = 38.0 + (wear_f * 40.0) + (v_rms * 0.3)
        
        # 3. Chaos Engine (Eventi stocastici di picco termico: 3% probabilità)
        chaos_mask = np.random.random(local_n) < 0.03
        local_temp[chaos_mask] += 15.0

    return local_temp

def main():
    # --- 1. INIZIALIZZAZIONE MPI ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    config = None
    global_health = None
    global_cycles = None
    global_temp_results = None

    # --- 2. GESTIONE CONFIGURAZIONE (Solo Rank 0) ---
    if rank == 0:
        print(f"--- Avvio Simulazione con {size} Processi MPI ---")
        config = load_or_create_config(FILE_CONFIG)
        print(f"Parametri di lancio: {config}")

    # --- 3. BROADCAST (bcast - minuscolo per oggetti Python) ---
    config = comm.bcast(config, root=0)
    
    num_pumps = int(config["num_pumps"])

    # Setup del partizionamento di memoria
    counts, displacements = calculate_partitions(num_pumps, size)
    local_n = counts[rank]

    # --- 4. ALLOCAZIONE ARRAY (Solo Rank 0 e buffer locali) ---
    if rank == 0:
        # Il master genera l'array globale delle pompe perfette
        global_health = np.full(num_pumps, 100.0, dtype=np.float64)
        global_cycles = np.zeros(num_pumps, dtype=np.float64)
        global_temp_results = np.empty(num_pumps, dtype=np.float64)

    # Tutti i processi allocano solo lo spazio strettamente necessario
    local_health = np.empty(local_n, dtype=np.float64)
    local_cycles = np.empty(local_n, dtype=np.float64)

    # Inizio misurazione performance
    start_time = MPI.Wtime()

    # --- 5. DISTRIBUZIONE DATI (Scatterv - MAIUSCOLO per buffer memory C-like) ---
    # Usando i metodi uppercase evitiamo l'overhead di serializzazione Pickle
    comm.Scatterv([global_health, counts, displacements, MPI.DOUBLE], local_health, root=0)
    comm.Scatterv([global_cycles, counts, displacements, MPI.DOUBLE], local_cycles, root=0)

    # --- 6. FASE DI CALCOLO PARALLELO ---
    local_results = local_simulation(local_health, local_cycles, rank, local_n, config)

    # --- 7. RACCOLTA DATI (Gatherv - MAIUSCOLO) ---
    # Riportiamo il dominio decomposto all'interno dell'array globale sul Master
    comm.Gatherv(local_results, [global_temp_results, counts, displacements, MPI.DOUBLE], root=0)

    # Barriera di Sincronizzazione: assicuriamoci che tutti abbiano finito prima di fermare il tempo
    comm.Barrier() 
    end_time = MPI.Wtime()

    # --- 8. POST-PROCESSING & OUTPUT (Solo Rank 0) ---
    if rank == 0:
        np.save(FILE_OUTPUT, global_temp_results)
        mean_temp = np.mean(global_temp_results)
        
        print("\n--- RISULTATI SIMULAZIONE ---")
        print(f"Totale pompe simulate         : {num_pumps}")
        print(f"Temperatura Media Globale     : {mean_temp:.2f} °C")
        print(f"Tempo di Esecuzione MPI       : {end_time - start_time:.4f} secondi")
        print(f"Risultati salvati su disco    : {FILE_OUTPUT}")

if __name__ == "__main__":
    main()