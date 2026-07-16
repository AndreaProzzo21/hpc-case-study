import os
import numpy as np
from mpi4py import MPI


CONFIG_FILE = "config.txt"


def read_config(filename):
    """
    Funzione per leggere i parametri da file. 
    Questa funzione verrà eseguita solo dal rank 0
    """
    
    # Crea un dizionario Python con i valori di base (default). 
    # Se il file non esiste, il programma userà questi.
    defaults = {
        "total_samples": 1000000,  # Numero totale di punti del segnale
        "window_size": 1000,       # Quanti punti passati guardare per calcolare la media locale
        "threshold": 5.0           # Se il segnale è tot volte la media, è un'anomalia
    }
    
    # Controlla se il file specificato dal percorso 'filename' (config.txt) NON esiste sul disco.
    if not os.path.exists(filename):
        # Apre (o crea) il file in modalità scrittura ("w" = write). 'f' è il puntatore al file.
        with open(filename, "w") as f:
            # Itera su tutte le coppie chiave-valore del dizionario 'defaults'
            for key, val in defaults.items():
                # Scrive fisicamente nel file la stringa "chiave=valore" e va a capo ("\n")
                f.write(f"{key}={val}\n")
        # Restituisce il dizionario di default al programma principale ed esce dalla funzione.
        return defaults
    
    # Se il file invece esiste, prepariamo un dizionario vuoto per ospitare i parametri letti.
    config = {}
    
    # Apre il file in modalità lettura ("r" = read).
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            
            if "=" in line and not line.startswith("#"):
                # .split("=") divide la stringa in due pezzi. 
                # Es: "total_samples=1000" diventa key="total_samples" e val="1000"
                key, val = line.split("=")
                
                if "." in val:
                    config[key.strip()] = float(val.strip())
                else:
                    config[key.strip()] = int(val.strip())
                    
    return config


def get_partitions(total_len, size):
    """
    Calcola quanti elementi spettano a ogni processo e da quale indice di memoria iniziano.
    """
    
    # np.full crea un array lungo 'size' (numero di processi).
    # Lo riempie con il risultato della divisione intera (//) tra elementi e processi.
    # Es: 10 // 3 = 3. L'array 'counts' sarà [3, 3, 3].
    counts = np.full(size, total_len // size, dtype=int)
    
    # Il modulo (%) trova il resto della divisione. Es: 10 % 3 = 1.
    # Questo slicing ([:resto]) seleziona i primi 'resto' processi e aggiunge 1 al loro carico.
    # Es: L'array diventa [4, 3, 3]. Ora la somma totale fa 10, e nessuno è sovraccarico!
    counts[:total_len % size] += 1
    
    # np.cumsum fa la somma cumulativa. Da [4, 3, 3] genera [4, 7, 10].
    # np.insert aggiunge uno '0' all'inizio: [0, 4, 7, 10].
    # [:-1] rimuove l'ultimo elemento. Risultato finale: [0, 4, 7].
    # Questi sono i displacements: il Rank 0 legge dall'indice 0, 
    # il Rank 1 dall'indice 4, il Rank 2 dall'indice 7.
    displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]
    
    # Restituisce le due tuple necessarie alle funzioni Scatterv e Gatherv.
    return counts, displacements


def process_signal(local_signal, window_size, threshold):
    """
    Analizza una parte di segnale = window size e cerca anomalie.
    Viene eseguito da ogni processo.
    """

    local_peaks = 0
    local_max = 0.0
    
    # Controlla se la fetta di dati ricevuta è più piccola della finestra di analisi.
    if len(local_signal) <= window_size:
        # Se l'array ha elementi, restituisce 0 picchi e il valore massimo dell'array. 
        # Altrimenti restituisce 0 e 0.0.
        return 0, np.max(local_signal) if len(local_signal) > 0 else 0.0

    # Inizia un ciclo FOR. Parte non da 0, ma dall'indice 'window_size' 
    # perché prima di quel punto non abbiamo abbastanza dati storici per fare una media completa.
    for i in range(window_size, len(local_signal)):
        
        # Se il valore del segnale nell'istante 'i' è maggiore del massimo storico registrato finora...
        if local_signal[i] > local_max:
            # aggiorna il massimo storico con questo nuovo valore.
            local_max = local_signal[i]
            
        # Crea una finestra dell'array che va dall'indice (i - window_size) fino a 'i' escluso.
        window = local_signal[i - window_size : i]
        
        # np.mean calcola la media aritmetica di questa finestra.
        local_mean = np.mean(window)
        
        if local_signal[i] > (local_mean * threshold):
            local_peaks += 1

    # Ritorna i due risultati: un numero intero (i picchi) e un decimale (il valore massimo).
    return local_peaks, local_max


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    # Deve esistere per tutti i processi prima di chiamare bcast, altrimenti Python dà errore "variabile non definita".
    config = None
 
    if rank == 0:
        config = read_config(CONFIG_FILE)
    
    config = comm.bcast(config, root=0)
    
    # Tutti i processi estraggono i valori dal dizionario.
    total_samples = int(config["total_samples"])
    window_size = int(config["window_size"])
    threshold = float(config["threshold"])

    # Tutti i processi chiamano la funzione per sapere esattamente come verrà diviso il mega-array.
    counts, displacements = get_partitions(total_samples, size)
    # 'local_n' è il numero di elementi che spetta al processo.
    local_n = counts[rank]

    global_signal = None
    
    if rank == 0:
        
        np.random.seed(123)
        # Genera un vettore di lunghezza 'total_samples' pieno di numeri casuali tra 0 e 1.
        # .astype(np.float64) forza il tipo a doppia precisione (8 byte).
        global_signal = np.random.rand(total_samples).astype(np.float64)
        
        # Iniettiamo 50 spikes.
        num_fake_spikes = 50
        # Genera 50 indici (posizioni nell'array) casuali compresi tra 0 e total_samples.
        spikes_indices = np.random.randint(0, total_samples, num_fake_spikes)
        # Va a quelle 50 posizioni esatte nell'array e somma 15.0 al loro valore attuale, creando picchi abnormi.
        global_signal[spikes_indices] += 15.0


    local_signal = np.empty(local_n, dtype=np.float64)


    comm.Barrier()
    start_time_global = 0.0
    if rank == 0:
        start_time_global = MPI.Wtime()

    comm.Scatterv([global_signal, counts, displacements, MPI.DOUBLE], local_signal, root=0)

    local_peaks, local_max = process_signal(local_signal, window_size, threshold)

    global_peaks = comm.reduce(local_peaks, op=MPI.SUM, root=0)
    global_max = comm.reduce(local_max, op=MPI.MAX, root=0)

    local_stats = np.array([rank, local_peaks, local_n], dtype=np.float64)
    
    global_stats = None
    if rank == 0:
        # Dimensione = (numero di processi * 3 elementi per processo)
        global_stats = np.empty(size * 3, dtype=np.float64)
        
    comm.Gather(local_stats, global_stats, root=0)

    if rank == 0:
        end_time_global = MPI.Wtime()
        total_simulation_time = end_time_global - start_time_global

        # Riformattiamo l'array 1D in una matrice (Numero_Processi x 3 colonne)
        stats_matrix = global_stats.reshape((size, 3))
        
        print("\n--- RISULTATI ANALISI SEGNALE ---")
        print(f"Campioni analizzati      : {total_samples}")
        print(f"Window size              : {window_size}")
        print(f"Picchi totali trovati    : {global_peaks}")
        print(f"Picco max assoluto       : {global_max:.2f}")
        print(f"TEMPO TOTALE SIMULAZIONE : {total_simulation_time:.4f} sec")
        
        print("\n--- REPORT ANOMALIE PER CORE ---")
        print("Rank\tAnomalies\tSample Size")
        print("-" * 30)
        
        for i in range(size):
            r = int(stats_matrix[i, 0])
            p = int(stats_matrix[i, 1])
            c = int(stats_matrix[i, 2])
            print(f"{r}\t{p}\t\t{c}")

if __name__ == "__main__":
    main()
