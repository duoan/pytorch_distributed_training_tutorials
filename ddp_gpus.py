
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.model = DistributedDataParallel(model, device_ids=[gpu_id])
    
    def _run_batch(self, xs, ys):
        self.optimizer.zero_grad()
        output = self.model(xs)
        loss = F.cross_entropy(output, ys)
        loss.backward()
        self.optimizer.step()
        
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_dataloader))[0])
        print(f"[GPU: {self.gpu_id} Epoch: {epoch}, Batch size: {batch_size} | Steps {len(self.train_dataloader)}]")
        self.train_dataloader.sampler.set_epoch(epoch) # type: ignore
        for xs, ys in self.train_dataloader:
            xs = xs.to(self.gpu_id)
            ys = ys.to(self.gpu_id)
            self._run_batch(xs, ys)
    
    def train(self, max_epoch: int):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)
            
            
class MyTrainDataset(Dataset):
    
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]

def main(rank: int, world_size: int, max_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    
    train_dataset = MyTrainDataset(2048)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_dataset),
    )
    
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    
    trainer = Trainer(
        model=model, 
        gpu_id=rank, 
        optimizer=optimizer, 
        train_dataloader=train_dataloader
    )
    
    trainer.train(max_epochs)
    
    destroy_process_group()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--max_epochs', type=int, default=10, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size on each device')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.max_epochs, args.batch_size), nprocs=world_size)