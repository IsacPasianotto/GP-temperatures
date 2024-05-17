# imports

import time
import numpy as np
import pandas as pd
from dask_jobqueue import SLURMCluster
from distributed import Client
# from dask import delayed

import fvgp
from fvgp import GP

import src.kernel as kernels
import src.hyperparameters as hps


def main():

    ### Constants ###

    INPUT_SPACE_DIM = 3
    N1 = 1
    N2 = 1
    DT = np.float64
    DATA_DIR = "data/"
    DATA_FILE_NAME = 'data_1960.csv'
    OUT_DIR = "out/"
    # MAT_SIZE = 10000, BATCH_SIZE = 100, 2 epyc --> 46 min train, ~ 2 min init 
    MAT_SIZE = 45000
    BATCH_SIZE = 750
    MAX_ITER_TRAIN = 100
    ENV_TO_SOURCE = 'source /u/dssc/ipasia00/test_dask/dask/bin/activate'

    ### Read the data ###

    data = pd.read_csv(DATA_DIR + DATA_FILE_NAME)
    data.dropna(inplace=True)
    x_data = data[['Latitude', 'Longitude', 'dt_float']].values
    y_data = data['AverageTemperature'].values

    idx = np.random.choice(np.arange(x_data.shape[0]), MAT_SIZE, replace=False)
    x_data = x_data[idx]
    y_data = y_data[idx]

    ### Set up the cluster ###

    cluster = SLURMCluster(cores=128,
                           memory="480GB",
                           processes=128,
                           job_cpu=128,
                           n_workers=1,
                           account="dssc",
                           queue="EPYC",
                           walltime="36:00:00",
                           job_script_prologue=['#SBATCH --output=' + OUT_DIR + '/slurm-%j.out',
                                '#SBATCH --job-name="GP-slave"',
                                'echo "-----------------------------------------------"',
                                'echo "HOSTNAME:           $(hostname)"',
                                'echo "DATE:               $(date)"',
                                'echo "-----------------------------------------------"',
                                'source ' + ENV_TO_SOURCE]
                           )

    cluster.scale(384) # Automatically will take all it can take, < 4 if 4 is not available
    # cluster.scale(256) # Automatically will take all it can take, < 4 if 4 is not available
    client = Client(cluster)
    # wait for workers to be ready
    time.sleep(35)
    print("Client is up and running", flush=True)
    print(client, flush=True)

    ### Define the model ###

    init_hps = hps.build_hps(N1, N2)
    hps_bounds = hps.build_bounds(N1, N2)

    t_init_start = time.time()
    gp = GP(INPUT_SPACE_DIM, x_data, y_data,
            init_hyperparameters=init_hps,
            gp_kernel_function=kernels.custom_kernel_one_shot,
            gp2Scale=True, gp2Scale_dask_client=client,
            gp2Scale_batch_size=BATCH_SIZE,
            info=False)
    t_init_end = time.time()

    print("===========================================", flush=True)
    print("Matrix size: ", MAT_SIZE, flush=True)
    print("Batch size: ", BATCH_SIZE, flush=True)
    print("===========================================", flush=True)
    print("Initialization time: ", t_init_end - t_init_start, flush=True)
    print("===========================================", flush=True)

    t_train_start = time.time()
    gp.train(hyperparameter_bounds=hps_bounds, max_iter=MAX_ITER_TRAIN, method='global')
    # gp.train(hyperparameter_bounds=hps_bounds, max_iter=MAX_ITER_TRAIN, method='mcmc')
    t_train_end = time.time()

    print("Training time: ", t_train_end - t_train_start)

    #### Just for now, to be sure the model is actually trained ####

    logfile = './out/hyperparameters.txt'
    with open(logfile, 'w') as f:
        f.write("-----------------------------------------------\n")
        f.write(f"init hps: {init_hps}")
        f.write("\n-----------------------------------------------\n")
        f.write(f"optimized hps: {gp.hyperparameters}")
        f.write("\n-----------------------------------------------\n")
        f.close()


if __name__ == '__main__':
    main()
