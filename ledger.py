import json
import os
from block import * 
from utils import *

class Ledger():
    def __init__(self):
        self.blocks = {}
        self.chain = []
        self.alpha = 4
        self.depth = 0
    
    def update(self, ledger_path):
        with open(ledger_path, 'w') as f:
            json.dump({'ledger': [self.buildBlock(block) for block in self.chain]}, f)
    
    def buildBlock(self, block):
        if block.type == "SecurityBlock":
            return {
                'type': block.type,
                'pHash': block.pHash,
                'wHash': block.wHash,
                'timestamp': block.timestamp,
                'loss': block.loss,
                'hash': block.getBlockHash(),
                'id': block.id
            }
        elif block.type == "ArmorBlock":
            return {
                'type': block.type,
                'checkpoint': {'optimizer' : {'name': block.checkpoints['optimizer'].__class__.__name__, 
                                                'state': block.checkpoints['optimizer'].state_dict()},
                                'weights' : {key: value.numpy().tolist() for key, value in block.checkpoints['model'].state_dict().items()}},
                'pHash': block.pHash,
                'nonce': block.nonce,
                'loss': block.loss,
                'hash' : block.getBlockHash(),
                'id' : block.id
            }
        else:
            print("The block doesn't exist!")
        
    def load(self, ledger_path):
        self.blocks = {}
        self.chain = []
        self.blocks['ArmorBlock'] = []
        self.blocks['SecurityBlock'] = []
        if os.path.exists(ledger_path):
            with open(ledger_path, 'r') as f:
                data = json.load(f)
                for block_data in data["ledger"]:
                    self.getBlock(block_data)
    
    def getBlock(self, block_data):
        if block_data['type'] == 'SecurityBlock':
            block = SecurityBlock(pHash=block_data['pHash'],
                          wHash=block_data['wHash'],
                          id=block_data['id'],
                          loss=block_data['loss'])
            block.timestamp = block_data['timestamp']
            self.blocks[block.type].append(block)
            self.chain.append(block)
        elif block_data['type'] == 'ArmorBlock':
            block = ArmorBlock(pHash=block_data['pHash'],
                                checkpoints=self.load_checkpoints(block_data),
                                id=block_data['id'],
                                nonce=block_data['nonce'],
                          loss=block_data['loss'])
            self.blocks[block.type].append(block)
            self.chain.append(block)  
    
    def load_checkpoints(self, block_data):
        model = Model((2,)) # ça c'est un problème ! (il faut loader dynamiquement)
        model.load_state_dict({key: torch.tensor(value) for key, value in block_data['checkpoint']['weights'].items()})
        
        optimizer = setOptimizer(model, block_data['checkpoint']['optimizer']['name'], block_data['checkpoint']['optimizer']['state']['param_groups'][0]['lr'])
        optimizer.load_state_dict(block_data['checkpoint']['optimizer']['state'])
        return {'model': model, 'optimizer': optimizer}
