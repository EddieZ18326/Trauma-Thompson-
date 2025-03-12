import os
import copy
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule
from torchinfo import summary
import torch.nn as nn


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    print(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = ViLTransformerSS(_config)
    summary(model, row_settings=("depth", "ascii_only"))
    print(model)
    exp_name = f'{_config["exp_name"]}'
    print('-----------------')
    print(_config)
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision('medium')


    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    
    print(_config["num_gpus"])
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="cuda",
        strategy = "ddp",
        benchmark=True,
        deterministic=True,
        max_epochs= 100, #_config["max_epoch"] if max_steps is None else 1000,
        max_steps=_config["max_steps"],
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        # flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        # weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )
    # trainer = pl.Trainer(
    #     # gpus=_config["num_gpus"],
    #     devices=_config["num_gpus"],
    #     num_nodes=_config["num_nodes"],
    #     precision=_config["precision"],
    #     accelerator="cuda", # "gpu",
    #     # strategy="ddp",
    #     benchmark=True,
    #     #deterministic=True,
    #     max_epochs=_config["max_epoch"] if max_steps is None else 1000,
    #     max_steps=max_steps,
    #     callbacks=callbacks,
    #     logger=logger,
    #     #prepare_data_per_node=False,
    #     replace_sampler_ddp=False,
    #     accumulate_grad_batches=grad_steps,
    #     log_every_n_steps=10,
    #     #flush_logs_every_n_steps=10,
    #     resume_from_checkpoint=_config["resume_from"],
    #     #weights_summary="top",
    #     fast_dev_run=_config["fast_dev_run"],
    #     val_check_interval=_config["val_check_interval"],
    # )

    
    hs = 768
    vs = 16
    
    # model.vqa_classifier = nn.Sequential(
    #             nn.Linear(hs, hs * 2),
    #             nn.LayerNorm(hs * 2),
    #             nn.GELU(),
    #             nn.Linear(hs * 2, 16),
            # )
    model.vqa_classifier[3] = nn.Linear(1536, 3129)
    '''
    # layers = list(model.children())
    # layers[-5] = nn.Sequential(
    #             nn.Linear(hs, hs * 2),
    #             nn.LayerNorm(hs * 2),
    #             nn.GELU(),
    #             nn.Linear(hs * 2, vs),
    #         )
    # model = nn.Sequential(*layers)
    # # '''
    print(model.vqa_classifier[3])    
    if not _config["test_only"]:
        print(model.vqa_classifier[3])    
        print(dm)
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
