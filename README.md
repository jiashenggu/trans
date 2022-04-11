conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c huggingface transformers   
conda install pandas matplotlib
Model	MR	MRR	Hits@1	Hits@3	Hits@10	
TransE	1264	0.51	0.41	0.57	0.67	
Hyperparameters:
k: 350; epochs: 4000; eta: 30; loss: multiclass_nll; optimizer: adam; optimizer_params: lr: 0.0001; regularizer: LP; regularizer_params: lambda: 0.0001; p: 2; embedding_model_params: norm: 1; normalize_ent_emb: false; seed: 0; batches_count: 100;