from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role

role = 'arn:aws:iam::623788847192:role/SageMakerExecutionRole' # get_execution_role()

# Hub Model configuration. https://huggingface.co/models
hub = {
  'HF_MODEL_ID':'sshleifer/distilbart-cnn-12-6',
  'HF_TASK':'summarization'
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    transformers_version='4.17.0',
    pytorch_version='1.10.2',
    py_version='py38',
    env=hub,
    role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(initial_instance_count=1,instance_type="ml.m5.xlarge")