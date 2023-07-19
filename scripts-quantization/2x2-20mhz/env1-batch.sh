python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-28-224-norm_mse-epoch40.yaml --log log-q/env1_2x2-20mhz/htov_224-28-224-norm_mse-epoch40.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-56-224-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-20mhz/htov_224-56-224-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-28-224-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-20mhz/htov_224-28-224-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-14-224-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-20mhz/htov_224-14-224-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-7-224-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-20mhz/htov_224-7-224-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-28-224-norm_mse-epoch40_sim-pretrained.yaml --log log-q/env1_2x2-20mhz/htov_224-28-224-norm_mse-epoch40_sim-pretrained.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-28-224-norm_mse-epoch40_sim-pretrained_adam.yaml --log log-q/env1_2x2-20mhz/htov_224-28-224-norm_mse-epoch40_sim-pretrained_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-896-896-448-448-224-224-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-20mhz/htov_224-896-896-448-448-224-224-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-896-896-896-672-672-672-448-448-448-224-224-224-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-20mhz/htov_224-896-896-896-672-672-672-448-448-448-224-224-224-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-896-1792-1792-896-224-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-20mhz/htov_224-896-1792-1792-896-224-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

python3 beamforming_matrix_estimate.py --config configs/env1_2x2-20mhz/htov_224-1792-1792-896-896-448-448-224-224-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-20mhz/htov_224-1792-1792-896-896-448-448-224-224-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'


