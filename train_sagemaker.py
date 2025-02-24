from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        "s3://<bucket-name>/tensorboard", 
        container_local_output_config="opt/ml/output/tensorboard"
        )
    
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='training',
        role='my-new-role',
        framework_version='2.5.1',
        py_version='py311',
        instance_count=1,
        instance_type='ml.g5.xlarge',
        hyperparameters={
            "batch-size": 32,
            "epochs": 25
        },
        tensorboard_config = tensorboard_config
    )

    estimator.fit({
        'train': 's3://<bucket-name>/dataset/train',
        'validation': 's3://<bucket-name>/dataset/validation',
        'test': 's3://<bucket-name>/dataset/test'
    })
if __name__ == "__main__":
    start_training()