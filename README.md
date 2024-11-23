# hierarchical-protein-classification

## Training
Before training, make sure your `.env` file is setup and `proteinworkshop` exists.
For training to work, line 400 of `proteinworkshop.models.base.py` needs to be changed to `self.log_dict(log_dict, prog_bar=True, batch_size=batch.batch_size)`.

Training is run through the command `python3 proteinvirtual/train.py encoder=<**encoder_name**> task=multiple_graph_classification dataset=fold_fold dataset.datamodule.num_workers=8 features=<**feature_set**> trainer=gpu scheduler=plateau trainer.max_epochs=150 optimiser.optimizer.lr=0.001 decoder.graph_label.dropout=0.5 ++test=True logger=wandb name=<**run_name**>
- The encoder name can be any **config** file in both `proteinvirtual/config` and `proteinworkshop/config`, but make sure that any new configs created have distinct names to prevent overwriting.
- The featuriser name should generally match with the encoder (i.e., **schnet** with **ca_bb** and **virtual_schnet_hierarchy** with **geo_hetero_hierarchy**)
- If using **wandb** as a logger, make sure your **wandb** information is setup in your `.env` file

## Testing
`pytest` is used for running tests, which can be added using `pip`.
To run tests, use the command `python3 -m pytest tests/`.

For the tests to work, make sure to create your own `.env` file from the included `.env.example` and link the path to your `proteinworkshop` folder for Hydra configs.
