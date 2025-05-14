# python generate_training_data.py --ec_label "4.2.1.1" --use_pivot --n_continuations 32 --n_batches 64 --model_path brenda_sft_stability_tuned_zym_control/epoch4.ckpt --tag "dpo0_high" --control_tag "<stability=high>"

# python generate_training_data.py --ec_label "4.2.1.1" --use_pivot --n_continuations 32 --n_batches 64 --model_path brenda_sft_stability_tuned_zym_control/epoch4.ckpt --tag "dpo0_medium" --control_tag "<stability=medium>"

# python generate_training_data.py --ec_label "4.2.1.1" --use_pivot --n_continuations 32 --n_batches 64 --model_path brenda_sft_stability_tuned_zym_control/epoch4.ckpt --tag "dpo0_low" --control_tag "<stability=low>"

python generate_training_data.py --ec_label "4.2.1.1" --use_pivot --n_continuations 32 --n_batches 64 --model_path brenda_sft_stability_tuned_zym_control/epoch4.ckpt --tag "dpo0_medium" --control_tag "<stability=medium>" --sequence_path /afs/cs.stanford.edu/u/waitz/waitz/prot-rl/DPO_ZymCTRL/train_data_iteration0_dpo0_medium/sequences_stability=medium.json

python generate_training_data.py --ec_label "4.2.1.1" --use_pivot --n_continuations 32 --n_batches 64 --model_path brenda_sft_stability_tuned_zym_control/epoch4.ckpt --tag "dpo0_low" --control_tag "<stability=low>" --sequence_path /afs/cs.stanford.edu/u/waitz/waitz/prot-rl/DPO_ZymCTRL/train_data_iteration0_dpo0_low/sequences_stability=low.json