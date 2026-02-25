wandb_project=factory-peginsert
for seed in 0 1 2 3 4 5
do
    python scripts/rl_games/evaluate_async.py --task Custom-Factory-PegInsert-Direct-UDR-v0 --mode all --num_envs 1024 --eval_episodes 10 --headless --track --seed $seed --wandb-project-name $wandb_project
done 