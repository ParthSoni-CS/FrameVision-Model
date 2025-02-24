from sagemaker.pytorch import PyTorch, PyTorchModel
import sagemaker

def deploy_endpoint():
    sagemaker.Session()
    role = "iam_role_arn"
    model_uri = "s3_bucket_path"

    
    model = PyTorchModel(model_data=model_uri,
                  role=role,
                  entry_point='inference.py',
                  framework_version='2.5.1',
                  py_version='py311',
                  source_dir='.',
                  name='framevision_model')  
    predictor = model.deploy(instance_type='ml.g5.xlarge', initial_instance_count=1, name = 'framevision_endpoint')  




if __name__ == "__main__":
    deploy_endpoint()