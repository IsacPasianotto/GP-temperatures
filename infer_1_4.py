# imports:
import time
import numpy as np
import pandas as pd
from dask_jobqueue import SLURMCluster
from distributed import Client
from fvgp import GP
import src.kernelonefour as kernels
import src.hyperparameters as hps
import sklearn.metrics as metrics

### Constants ###
INPUT_SPACE_DIM = 3
N1 = 1
N2 = 4
HPS_FILE = "./out/one_four.npy"
RESULT_FILE = "./out/inf_one_four.txt"

DT = np.float64
DATA_DIR = "./data/"
DATA_FILE_NAME = 'data_1960.csv'
MAT_SIZE = 75000
BATCH_SIZE = 2500
MAX_ITER_TRAIN = 10
TEST_SIZE = 1000  # due to a "bug" it has to be <= MAT_SIZE
OUT_DIR = './out'
ENV_TO_SOURCE = 'source /u/dssc/ipasia00/test_dask/dask/bin/activate'


def main():
    ## Read the data ###

    data = pd.read_csv(DATA_DIR + DATA_FILE_NAME)
    data.dropna(inplace=True)
    x_data = data[['Latitude', 'Longitude', 'dt_float']].values
    y_data = data['AverageTemperature'].values

    idx = np.random.choice(np.arange(x_data.shape[0]), MAT_SIZE, replace=False)
    x_train = x_data[idx]
    y_train = y_data[idx]

    x_eligible = np.delete(x_data, idx, axis=0)
    y_eligible = np.delete(y_data, idx, axis=0)

    new_idx = np.random.choice(np.arange(x_eligible.shape[0]), TEST_SIZE, replace=False)
    x_test = x_eligible[new_idx]
    y_test = y_eligible[new_idx]

    ### Set up the cluster ###

    cluster = SLURMCluster(cores=128,
                           memory="480GB",
                           processes=128,
                           job_cpu=128,
                           n_workers=0,
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
    # wait for workers to be ready
    time.sleep(35)

    client = Client(cluster)
    print("Client is up and running", flush=True)
    print(client, flush=True)

    ### Define the model ###

    init_hps = np.load(HPS_FILE)

    t_init_start = time.time()
    gp = GP(INPUT_SPACE_DIM, x_train, y_train,
            init_hyperparameters=init_hps,
            gp_kernel_function=kernels.custom_kernel_one_shot,
            gp2Scale=True, gp2Scale_dask_client=client,
            gp2Scale_batch_size=BATCH_SIZE,
            info=False)
    t_init_end = time.time()

    # print the info to the file:
    with open(RESULT_FILE, 'w') as f:
        print("===========================================", file=f)
        print("N1: ", N1, file=f)
        print("N2: ", N2, file=f)
        print("...........................", file=f)
        print("Matrix size: ", MAT_SIZE, file=f)
        print("Batch size: ", BATCH_SIZE, file=f)
        print("Test size: ", TEST_SIZE, file=f)
        print("===========================================", file=f)
        print("Initialization time: ", t_init_end - t_init_start, file=f)
        print("===========================================", file=f)


    #### Perform the prediction ###
    t_infer_start = time.time()
    y_pred = gp.posterior_mean(x_test)["f(x)"]
    t_infer_end = time.time()

    with open(RESULT_FILE, 'a') as f:
        print("Inference time: ", t_infer_end - t_infer_start, file=f)

    # evaluate the model
    mse = metrics.mean_squared_error(y_test, y_pred)

    with open(RESULT_FILE, 'a') as f:
        print("MSE: ", mse, file=f)
        print("===========================================", file=f)

if __name__ == '__main__':
    main()
