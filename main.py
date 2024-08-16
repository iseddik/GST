import threading
import matplotlib.pyplot as plt
from trainer import Trainer
from utils import *

class TrainingSession:
    def __init__(self, lr=0.01, epochs=10):
        self.training_args = {
            "criterion": setCriterion('BCELoss'),
            "lr": lr,
            "epochs": epochs,
            "optimizer_name": 'SGD'
        }
        self.node = Trainer(0, self.training_args["criterion"])
        self.model = Model((2,))
        self.setup_model()

    def setup_model(self):
        # Uncomment the following line when ready to setup the model
        # self.node.setupWorkNet(model=self.model, trainingArguments=self.training_args)
        self.node.ledger.load(ledger_path="ledger.dat")

    def launch_training(self):
        print("Starting training session...")
        self.node.train(self.training_args)
        print("Training session completed.")

    def sync_ledger(self):
        self.node.syncLedger()

    def show_loss_graph(self):
        self.node.ledger.load(ledger_path="ledger.dat")
        loss = [b.loss for b in self.node.ledger.chain[1:]]  # Exclude first element
        ids = [b.id for b in self.node.ledger.chain[1:]]  # Get the IDs
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Line plot
        ax1.plot(range(len(loss)), loss)
        ax1.set_title("Training Loss Over Time")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        
        # Scatter plot
        colors = ['red', 'green', 'blue']
        for i in range(3):
            mask = [id == i for id in ids]
            ax2.scatter([j for j, m in enumerate(mask) if m], 
                        [l for l, m in zip(loss, mask) if m], 
                        c=colors[i], label=f'ID {i}')
        
        ax2.set_title("Training Loss Over Time (Scatter)")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('loss.png')
        plt.close()
        print("Loss graph saved as 'loss.png'")

    def start_listening(self):
        start_listening(self.node)

class UserInterface:
    def __init__(self, training_session):
        self.training_session = training_session

    def run(self):
        commands = {
            'yes': self.training_session.launch_training,
            'sync': self.training_session.sync_ledger,
            'show': self.training_session.show_loss_graph,
        }

        while True:
            user_input = input("Enter command (yes/no/exit/sync/show): ").strip().lower()
            if user_input == 'exit':
                print("Exiting the program.")
                break
            elif user_input in commands:
                commands[user_input]()
            elif user_input == 'no':
                print("Training session not launched.")
            else:
                print("Invalid input. Please enter a valid command.")

def main():
    training_session = TrainingSession()
    
    listening_thread = threading.Thread(target=training_session.start_listening)
    listening_thread.daemon = True
    listening_thread.start()

    ui = UserInterface(training_session)
    ui.run()

if __name__ == "__main__":
    main()
