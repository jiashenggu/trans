from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.datasets import YAGO310
from pykeen.models import predict
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.regularizers import LpRegularizer
from torch.optim import Adam
import torch


TransE_50 = torch.load('/nas/home/gujiashe/trans/YAGO310_d350_b128_epochs50/trained_model.pkl')

# Run the pipeline
pipeline_result = pipeline(
    model=TransE_50,
    dataset=YAGO310,
    evaluator='RankBasedEvaluator',
    training_loop='sLCWA',
    negative_sampler='basic',
    model_kwargs=dict(
        scoring_fct_norm=2,
        embedding_dim=350,
    ),
    optimizer=Adam,
    optimizer_kwargs=dict(lr=1.0e-4),
    training_kwargs=dict(num_epochs=200, batch_size=256),
    regularizer = LpRegularizer(weight=1e-4, p=2),
    random_seed = 0,
)
# save the model
pipeline_result.save_to_directory('YAGO310_d350_b256_epochs200')
