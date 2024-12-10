import socket

class Path:
    
    @staticmethod
    def db_root_dir(dataset='TACO'):
        return f'./data/{dataset}'

    @staticmethod
    def models_dir(arch='detector'):
        return f"./model'/{arch}"
            

    @staticmethod
    def root_models_dir():
        return f'./model'
        
   