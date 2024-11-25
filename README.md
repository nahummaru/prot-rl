
# DPO_pLM
You can see more information about DPO in this our preprint: 
## Installation

To set up the environment, install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running Experiments

To reproduce our experiments, you can execute the `.sh` files corresponding to each experiment located in the `experiments` folder. For example:

```bash
bash experiments/experiment_name.sh
```

Replace `experiment_name` with the desired experiment script.

## Testing New Features

If you are interested in testing other features, you can use the provided template as a starting point for your custom experiments.

## About DPO_pLM

`DPO_pLM` is the main Python script for our project. It supports the following flags:

- `--beta`: Specify the beta value.
- `--mode`: Choose the mode for experiments. Available options:
  - `paired`
  - `ranked`
  - `weighted`

Refer to the documentation or comments in the script for more details on how to configure and run your experiments.

Feel free to contribute or raise issues if you encounter any problems!
