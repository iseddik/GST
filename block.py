import datetime
import hashlib
from utils import getWeightsHash 

class SecurityBlock:
    def __init__(self, wHash, pHash, id, loss=None):
        self.type = "SecurityBlock"
        self.pHash = pHash
        self.wHash = wHash
        self.loss = loss
        self.id = id
        self.timestamp = str(datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()))

    def getBlockHash(self):
        merkle = str(self.wHash) + str(self.pHash) + str(self.timestamp)
        return hashlib.sha256(merkle.encode()).hexdigest()
    

class ArmorBlock: 
    def __init__(self, checkpoints, pHash, nonce, id, loss=None):
        self.type = "ArmorBlock"
        self.checkpoints = checkpoints 
        self.pHash = pHash
        self.nonce = nonce
        self.id = id
        self.loss = loss
    
    def getBlockHash(self):
        merkle = str(getWeightsHash(self.checkpoints['model'])) + str(self.pHash) + str(self.nonce)
        return hashlib.sha256(merkle.encode()).hexdigest()