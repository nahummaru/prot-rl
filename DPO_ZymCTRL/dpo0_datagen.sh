python generate_training_data.py --ec_label "4.2.1.1" --use_pivot --n_continuations 32 --n_batches 1000 --model_path brenda_sft_stability_tuned_zym_control/epoch4.ckpt --tag "test1" --control_tag "<stability=high>"

python generate_training_data.py --ec_label "4.2.1.1" --use_pivot --n_continuations 32 --n_batches 1000 --model_path brenda_sft_stability_tuned_zym_control/epoch4.ckpt --tag "test2" --control_tag "<stability=medium>"

python generate_training_data.py --ec_label "4.2.1.1" --use_pivot --n_continuations 32 --n_batches 1000 --model_path brenda_sft_stability_tuned_zym_control/epoch4.ckpt --tag "test3" --control_tag "<stability=low>"