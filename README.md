# [2024-1 Deeplearning] Term project POSTECH
# Project Title

## Environment Setting

1. Create an environment with Anaconda:
    ```bash
    conda create -n team6 python=3.10
    ```

2. Install all the necessary packages by running the following command in the parent directory:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the final package:
    ```bash
    conda install mpi4py
    ```

## Model Training

1. Install `mnist1d`:
    ```bash
    pip install mnist1d
    ```

2. See the “#Settings” part for hyperparameter configurations. We primarily experimented with varying depth, but other parameters are also adjustable.

3. Change directory to `mnist1d`:
    ```bash
    cd ../mnist1d
    ```

4. Run the training script:
    ```bash
    python he_test_by_depth_final.py -depth=n
    ```
    Replace `n` with the desired depth value.

5. This code can be resumed since the log is saved in `wandb` and the model is also saved in a local checkpoint. It also saves intermediate values for the metrics being calculated and the loss trajectory.

## Calculating Metrics

1. You need a CSV file with convergence steps and loss before and after training by depth, as well as paths with weight values before and after training.

2. If the paths are specified correctly, run the Jupyter notebook file to calculate the metric values and save the results.

3. Perform the correlation test by fetching the required columns from the results. Columns 8 and 9 store the column-specific correlation coefficient values with depth. Column 8 shows the Spearman test coefficient and column 9 shows the Pearson test coefficient with p-value.

## PyHessian

1. Install `pyhessian`:
    ```bash
    pip install pyhessian
    ```

2. Download `pyhessian` from GitHub.

3. Change directory to `PyHessian`:
    ```bash
    cd PyHessian
    ```

4. Run the `loss_landscape_visualization.ipynb` notebook.

5. (Please load the model according to the description in the markdown)
    1. Load the model with the matching depth.
    2. Ensure the model path is correctly set.

## Loss Trajectory

1. Navigate to the `loss-landscape` folder:
    ```bash
    cd loss-landscape
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Plot the surface:
    ```bash
    python plot_surface.py --cuda --model fcnn --x=-1:1:51 --y=-1:1:51 --model_file /home/yeonjoo/deepl/weight/depth1/w__depth_1.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --depth 1 --raw_data
    ```
    - Adding `mpirun -n 4` enables parallel processing, but it might not work in some cases, so it is set to default as excluded.
    - `x` and `y`: Range to search for the loss landscape. The default `-1:1:51` samples 51 points between -1 and 1 for each point.
    - `model_file`: Path to the pre-trained model.
    - `depth`: The current depth to experiment with.

4. Plot the trajectory:
    ```bash
    python plot_trajectory.py --dataset mnist1d --model fcnn --model_folder model_save_folder --start_epoch 0 --max_epoch 300 --save_epoch 10 --prefix model_ --suffix .pth --depth 1
    ```
    - Requires checkpoints saved as `model_{epoch}.pth`. Default saves every 10 epochs.
    - `model_folder`: Folder where models are saved at each `save_epoch`.
    - `start_epoch`: Starting epoch (default: 0).
    - `max_epoch`: Last saved epoch (varies by model).
    - `save_epoch`: Saving interval (default: 10).
    - `prefix`: Prefix for saved models (default: model_).
    - `suffix`: Suffix for saved models (default: .pth).
    - `depth`: Current depth to experiment with.

5. Plot 2D:
    ```bash
    python plot_2D.py --surf_file /home/yeonjoo/deepl/weight/w__depth_3.pth_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,51]x[-1.0,1.0,51].h5 --dir_file /model_folder/PCA_directions.h5 --proj_file /model_folder/PCA_proj_cos.h5
    ```
    - `surf_file`: The h5 file generated after running `plot_surface.py`, e.g., `/home/yeonjoo/deepl/weight/w__depth_3.pth_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,51]x[-1.0,1.0,51].h5`.
    - `dir_file`: Path to the `directions.h5` file in the `PCA_*` folder generated when running `plot_trajectory.py`.
    - `proj_file`: Path to the `*_proj_cos.h5` file in the `PCA_*` folder generated when running `plot_trajectory.py`.

6. Code Reference:
   - MNIST1D:
     https://github.com/greydanus/mnist1d
   - PyHessian:
     https://github.com/amirgholami/PyHessian
   - Loss-landscape:
     https://github.com/tomgoldstein/loss-landscape
