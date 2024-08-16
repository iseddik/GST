from ledger import Ledger
from utils import *
from copy import deepcopy
from block import *
import socket
import pickle

PORT = 5000
BUFFER_SIZE = 1024


class Trainer():
    network = ['172.19.0.3', '172.19.0.4']

    def __init__(self, id, criterion):
        self.ledger = Ledger()
        self.id = id
        self.privateKey = getPrivate()
        self.publicKey = getPublic(self.privateKey)
        self.ledger_path = "ledger.dat"
        self.queue = []
        self.criterion = criterion
        self.train_loader = load_train_data('train.pth')
        self.audit_loader = load_test_data('audit.pth')
        self.model = None

    def train(self, trainingArguments):
        self.ledger.load(ledger_path="ledger.dat")
        self.model = deepcopy(self.ledger.blocks["ArmorBlock"][-1].checkpoints['model'])
        optimizer = setOptimizer(self.model, "SGD", 0.1)
        optimizer.load_state_dict(self.ledger.blocks["ArmorBlock"][-1].checkpoints['optimizer'].state_dict())
        for i in range(trainingArguments['epochs']):
            total_loss = self.go(trainingArguments['criterion'], optimizer)
            print(f'Epoch {i+1}, Loss: {total_loss/len(self.train_loader)}')
    
    def go(self, criterion, optimizer):
        if len(self.queue) == 4:
            self.forgot()
            self.syncLedger()
        return self.step(criterion, optimizer)
    
    def step(self, criterion, optimizer):
        total_loss = 0
        seq = self.fork() 
        lo = 100 if self.ledger.chain[-1].loss == None else self.ledger.chain[-1].loss
        checkp = self.preparCheckpoint(self.model, optimizer)
        for inputs, targets in self.train_loader:
            optimizer.zero_grad()  
            outputs = self.model(inputs)  
            loss = criterion(outputs, targets.unsqueeze(1))  # la il faut intégrer dataset d'audit
            loss.backward()  
            optimizer.step()
            out = self.model(self.audit_loader.dataset.tensors[0])  
            los = criterion(out, self.audit_loader.dataset.tensors[1].unsqueeze(1))
            if los.item() < lo:
                lo = los.item()
                block = SecurityBlock(wHash=getWeightsHash(self.model),
                                pHash=self.ledger.blocks["SecurityBlock"][-1].getBlockHash(),
                                id=self.id,
                                loss=lo)
                self.ledger.blocks[block.type].append(block)
                self.ledger.chain.append(block)
                seq.append(block)
                model_state = deepcopy(self.model)
                optimizer_state = deepcopy(optimizer)
                checkp = self.preparCheckpoint(model_state, optimizer_state)
            total_loss += loss.item()
        hash = str(getWeightsHash(checkp['model'])) + str(self.ledger.chain[-1].getBlockHash())
        nonce,_ = proof_of_work(hash, self.ledger.alpha)
        armor_block = ArmorBlock(pHash=self.ledger.blocks["SecurityBlock"][-1].getBlockHash(),
                                checkpoints=checkp,
                                id=self.id,
                                nonce=nonce,
                                loss=lo)
        self.ledger.blocks[armor_block.type].append(armor_block)
        self.ledger.chain.append(armor_block)
        seq.append(armor_block)
        broadcast(self.network, (seq, self.ledger.depth+1))
        self.queue.append((seq, self.ledger.depth+1))
        if len(self.queue) == 4:
            self.syncLedger()
        return total_loss
    
    def preparCheckpoint(self, model, optimizer):
        return {'model': model, 'optimizer': optimizer}
    
    def fork(self):
        securityBlocks = []
        block = SecurityBlock(wHash=getWeightsHash(self.ledger.blocks["ArmorBlock"][-1].checkpoints['model']),
                             pHash=self.ledger.blocks["SecurityBlock"][-1].getBlockHash(),
                             id=self.id,
                             loss = self.ledger.blocks["ArmorBlock"][-1].loss)
        securityBlocks.append(block)
        self.ledger.blocks[block.type].append(block)
        self.ledger.chain.append(block)
        return securityBlocks

    def isValidSeq(self, seq, criterion): 
        self.ledger.load(ledger_path="ledger.dat")
        vergBlock = self.ledger.blocks['SecurityBlock'][-1]
        armorBlock = self.ledger.blocks['ArmorBlock'][-1]
        if vergBlock.getBlockHash() != seq[0][0].pHash:
            return False
        elif  armorBlock.loss != None and armorBlock.loss <= self.getScore(seq[0][-1], criterion).item() : 
            return False
        elif countZeros(seq[0][-1].getBlockHash()) < self.ledger.alpha:
            return False
        return True
    
    def getScore(self, block, criterion):
        model = block.checkpoints['model']
        out = model(self.audit_loader.dataset.tensors[0])  
        loss = criterion(out, self.audit_loader.dataset.tensors[1].unsqueeze(1))
        return loss

    def getBestSeq(self, list):
        seq = list[0]
        for s in list:
            if s[0][-1].loss < seq[0][-1].loss:
                seq = s
        return seq

    def forgot(self):
        if len(self.ledger.blocks['ArmorBlock']) == 3:
            obj = self.ledger.blocks['ArmorBlock'][0]
            self.ledger.blocks['ArmorBlock'].pop(0)
            self.ledger.chain.remove(obj)
        
    def syncLedger(self):
        if self.queue != []:
            depths = [s[1] for s in self.queue]
            depths.sort()
            for d in depths:
                condidateSeqs = []
                for seq in self.queue:
                    if self.isValidSeq(seq, self.criterion):
                        condidateSeqs.append(seq)
                if condidateSeqs != []:
                    seq = self.getBestSeq(condidateSeqs)
                    self.ledger.chain += seq[0]
                    self.ledger.blocks['SecurityBlock'] += seq[0][:-1]
                    self.ledger.blocks['ArmorBlock'] += [seq[0][-1]]
                    self.ledger.depth = d
                    self.forgot()  
                    self.ledger.update(ledger_path=self.ledger_path)  
                else:
                    print("No sync in depth -> ", d)
        self.queue = []
