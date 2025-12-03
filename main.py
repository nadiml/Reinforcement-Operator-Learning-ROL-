import os
import numpy as np
import torch
import seaborn as sns

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor


from config import (
    SEED,
    KS_L,
    KS_NX,
    KS_NU,
    KS_DT,
    DEEPN_HIDDEN_DIM,
    DEEPN_DROPOUT,
    DEEPN_LR,
    DEEPN_WEIGHT_DECAY,
    DEEPN_EPOCHS,
    DEEPN_BATCH_SIZE,
    DEEPN_NUM_TRAJ,
    DEEPN_T_MAX,
    DEEPN_NOISE_LEVEL,
    RL_MAX_STEPS,
    RL_TOTAL_TIMESTEPS_PURE,
    RL_TOTAL_TIMESTEPS_RESIDUAL,
    RL_BUFFER_SIZE,
    RL_GAMMA,
    RL_TAU,
    RL_BATCH_SIZE,
    RL_LR_PURE,
    RL_LR_RESIDUAL,
    PURE_NOISE_SIGMA,
    RES_NOISE_SIGMA,
    EVAL_STEPS,
    EVAL_N_TRIALS,
    CHECK_FREQ,
    SMOOTH_WINDOW,
)

from src.ks_system import KSSolver, LQRBaseline
from src.deeponet_operator import (
    DeepONet,
    mpc_expert_policy,
    generate_deeponet_data,
    save_deeponet_dataset,
    train_deeponet_with_log,
)
from src.ks_envs import PureRLKSEnv, ResidualDeepONetKSEnv, SaveOnBestRewardCallback
from src.evaluation_plotting import (
    eval_policy,
    eval_lqr_baseline,
    plot_comprehensive_figures,
)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    sns.set_context("paper")
    sns.set_style("whitegrid")
    

    # solver setup-----------
    solver = KSSolver(L=KS_L, nx=KS_NX, nu=KS_NU, dt=KS_DT)
    nx = solver.nx
    steps = EVAL_STEPS
    n_trials = EVAL_N_TRIALS
    solver_x = solver.x
    solver_t = np.arange(steps) * solver.dt

    # Data Generate--------------
    branch_inputs, trunk_inputs, outputs = generate_deeponet_data(
        solver,
        mpc_expert_policy,
        num_trajectories=DEEPN_NUM_TRAJ,
        t_max=DEEPN_T_MAX,
        noise_level=DEEPN_NOISE_LEVEL,
    )
    save_deeponet_dataset(
        branch_inputs,
        trunk_inputs,
        outputs,
        path="deeponet_dataset.npz",
    )

    # DeepOnet Training -----------
    deeponet = DeepONet(
        branch_input_dim=nx,
        trunk_input_dim=1,
        hidden_dim=DEEPN_HIDDEN_DIM,
        output_dim=nx,
        dropout=DEEPN_DROPOUT,
    )
    deeponet, deeponet_loss_list = train_deeponet_with_log(
        deeponet,
        branch_inputs,
        trunk_inputs,
        outputs,
        epochs=DEEPN_EPOCHS,
        batch_size=DEEPN_BATCH_SIZE,
        lr=DEEPN_LR,
        weight_decay=DEEPN_WEIGHT_DECAY,
    )

    # Pure-RL Training
    if not os.path.exists("pure_rl_log"):
        os.makedirs("pure_rl_log")

    pure_env = Monitor(
        PureRLKSEnv(solver, max_steps=RL_MAX_STEPS),
        filename="pure_rl_log/monitor.csv",
    )
    venv_pure = DummyVecEnv([lambda: pure_env])
    venv_pure = VecNormalize(
        venv_pure, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    action_noise_pure = NormalActionNoise(
        mean=np.zeros(nx), sigma=PURE_NOISE_SIGMA * np.ones(nx)
    )

    callback_best_pure = SaveOnBestRewardCallback(
        check_freq=CHECK_FREQ,
        save_path="best_pure_rl_model",
        verbose=1,
        smooth_window=SMOOTH_WINDOW,
    )
    
    model_PureRL = TD3(
        "MlpPolicy",
        venv_pure,
        action_noise=action_noise_pure,
        verbose=1,
        buffer_size=RL_BUFFER_SIZE,
        seed=SEED,
        learning_rate=RL_LR_PURE,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        gamma=RL_GAMMA,
        tau=RL_TAU,
        batch_size=RL_BATCH_SIZE,
    )
    model_PureRL.learn(
        total_timesteps=RL_TOTAL_TIMESTEPS_PURE, callback=callback_best_pure
    )


    # DeepOnet Guioded Residual TD3
    if not os.path.exists("deeponet_rl_log"):
        os.makedirs("deeponet_rl_log")

    residual_env = Monitor(
        ResidualDeepONetKSEnv(solver, deeponet, max_steps=RL_MAX_STEPS),
        filename="deeponet_rl_log/monitor.csv",
    )
    venv_res = DummyVecEnv([lambda: residual_env])
    venv_res = VecNormalize(
        venv_res, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    action_noise_res = NormalActionNoise(
        mean=np.zeros(nx), sigma=RES_NOISE_SIGMA * np.ones(nx)
    )

    callback_best_res = SaveOnBestRewardCallback(
        check_freq=CHECK_FREQ,
        save_path="best_deeponet_rl_model",
        verbose=1,
        smooth_window=SMOOTH_WINDOW,
    )

    model_RL_DeepONet = TD3(
        "MlpPolicy",
        venv_res,
        action_noise=action_noise_res,
        verbose=1,
        buffer_size=RL_BUFFER_SIZE,
        seed=SEED + 1,
        learning_rate=RL_LR_RESIDUAL,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        gamma=RL_GAMMA,
        tau=RL_TAU,
        batch_size=RL_BATCH_SIZE,
    )
    model_RL_DeepONet.learn(
        total_timesteps=RL_TOTAL_TIMESTEPS_RESIDUAL, callback=callback_best_res
    )

    # LQR Baseline------
    if not os.path.exists("deeponet_rl_log"):
        os.makedirs("deeponet_rl_log")

    residual_env = Monitor(
        ResidualDeepONetKSEnv(solver, deeponet, max_steps=RL_MAX_STEPS),
        filename="deeponet_rl_log/monitor.csv",
    )
    venv_res = DummyVecEnv([lambda: residual_env])
    venv_res = VecNormalize(
        venv_res, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    action_noise_res = NormalActionNoise(
        mean=np.zeros(nx), sigma=RES_NOISE_SIGMA * np.ones(nx)
    )

    callback_best_res = SaveOnBestRewardCallback(
        check_freq=CHECK_FREQ,
        save_path="best_deeponet_rl_model",
        verbose=1,
        smooth_window=SMOOTH_WINDOW,
    )

    model_RL_DeepONet = TD3(
        "MlpPolicy",
        venv_res,
        action_noise=action_noise_res,
        verbose=1,
        buffer_size=RL_BUFFER_SIZE,
        seed=SEED + 1,
        learning_rate=RL_LR_RESIDUAL,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        gamma=RL_GAMMA,
        tau=RL_TAU,
        batch_size=RL_BATCH_SIZE,
    )
    model_RL_DeepONet.learn(
        total_timesteps=RL_TOTAL_TIMESTEPS_RESIDUAL, callback=callback_best_res
    )
    
    # LQR-Baseline 
    lqr_baseline = LQRBaseline(
        nx=solver.nx, L=solver.L, nu=solver.nu, Q=1.0, R=0.01
    )

    ### EValuation-------------
    eval_env_pure = PureRLKSEnv(solver, max_steps=steps)
    states_pure, energies_pure = eval_policy(
        eval_env_pure, model_PureRL, steps=steps, n_trials=n_trials
    )

    eval_env_res = ResidualDeepONetKSEnv(solver, deeponet, max_steps=steps)
    states_deeponet_rl, energies_deeponet_rl = eval_policy(
        eval_env_res, model_RL_DeepONet, steps=steps, n_trials=n_trials
    )

    eval_env_lqr = PureRLKSEnv(solver, max_steps=steps)
    states_lqr, energies_lqr = eval_lqr_baseline(
        eval_env_lqr, lqr_baseline, steps=steps, n_trials=n_trials
    )

    # Visualizatoin
    states_dict = {
        "Pure RL": states_pure,
        "DeepONet RL": states_deeponet_rl,
        "LQR Baseline": states_lqr,
    }
    energies_dict = {
        "Pure RL": energies_pure,
        "DeepONet RL": energies_deeponet_rl,
        "LQR Baseline": energies_lqr,
    }

    plot_comprehensive_figures(
        deeponet_loss_list=deeponet_loss_list,
        pure_rl_callback=callback_best_pure,
        deeponet_rl_callback=callback_best_res,
        states_dict=states_dict,
        energies_dict=energies_dict,
        solver_x=solver_x,
        solver_t=solver_t,
        save_dir="figures",
    )

if __name__=="__main__":
    main()