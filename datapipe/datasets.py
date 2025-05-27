from basicsr.data.holo_dataset import Dataset_Holo_256

def create_dataset(dataset_config):
    if dataset_config['type'] == 'holo_inverse':
        dataset = Dataset_Holo_256(dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])
    
    return dataset






