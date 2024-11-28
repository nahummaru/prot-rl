![image](https://github.com/user-attachments/assets/b3b0390b-d768-4b3f-97d1-5ad4c24a436d)![image](https://github.com/user-attachments/assets/31a95260-fd0b-4bf5-ac29-55dddc28491b)


# DPO_pLM
You can see more information about DPO in our preprint: 
## Installation

To set up the environment, install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
## Example 
Here we report a very simple script with the objective is to increase the lenght over the different iterations. In the 'Exeperiments' folder more sophysticated scoring function are considered. 
If you are interested in testing other features, you can use this template as a starting point for your custom experiments.

First of all you need to set up zymcontrol, using HF repository or the local, specifiying as a flag the directory of the model or the HF name (e.g AI4PD/ZymCTRL). 
In the folder you execute will be generated the sequences for each iteration
Reuslts: ![image](https://github.com/user-attachments/assets/b408b256-0697-45b2-a396-2312f87f1ed8)



## About DPO_pLM

`DPO_pLM` is the main Python script for our project. It supports the following flags:

- `--beta`: Specify the beta value.
- '--model_directory': Specify the local model or the HF name (e.g AI4PD/ZymCTRL or /your/path/ZymCTRL)
- `--mode`: Choose the mode for experiments. Available options:
  - `paired`
  - `ranked`
  - `weighted`

## Experiments Reprudicibility

To reproduce our experiments, you can execute the `.sh` files corresponding to each experiment located in the `Experiments` folder. For example:

```bash
bash experiments/experiment_name.sh
```

Replace `experiment_name` with the desired experiment script.

Each experiment will produce, fold and calulcate statistics for each considered feature. We additionaly add out script for plotting the different features as plotted in our manuscript.


Refer to the documentation or comments in the script for more details on how to configure and run your experiments.

Feel free to contribute or raise issues if you encounter any problems!
