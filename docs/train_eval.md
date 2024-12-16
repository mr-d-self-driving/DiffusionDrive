# DiffusionDrive Training and Evaluation

## 1. Cache dataset for faster training and evaluation
```bash
# cache dataset for training
python navsim/planning/script/run_dataset_caching.py agent=diffusiondrove_agent experiment_name=training_diffusiondrive_agent train_test_split=navtrain

# cache dataset for evaluation
python navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

## 2. Training
If your training machine does not have network access, you should download the pretrained ResNet-34 model from [huggingface](https://huggingface.co/timm/resnet34.a1_in1k) and upload it to your training machine.
```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=diffusiondrive_agent \
        experiment_name=training_diffusiondrive_agent  \
        train_test_split=navtrain  \
        split=trainval   \
        trainer.params.max_epochs=100 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 
```

## 3. Evaluation
You can use the following command to evaluate the trained model `export CKPT=/path/to/your/checkpoint.pth`, for example, you can download the our provided checkpoint from [huggingface](https://huggingface.co/hustvl/DiffusionDrive), and set `CKPT=/path/to/downloaded/huggingface_diffusiondrive_agent_ckpt/diffusiondrive_navsim_88p1_PDMS.pth`:
```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=diffusiondrive_agent \
        worker=ray_distributed \
        agent.checkpoint_path=$CKPT \
        experiment_name=diffusiondrive_agent_eval
```
