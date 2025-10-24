import asyncio

import torch

from monarch.actor import Actor, current_rank, endpoint

import torchstore as ts
from torchstore.utils import spawn_actors


WORLD_SIZE = 1


# In monarch, Actors are the way we represent multi-process/node applications. For additional details, see:
# https://github.com/meta-pytorch/monarch?tab=readme-ov-file#monarch-
class ExampleActor(Actor):
    def __init__(self, world_size=WORLD_SIZE):
        self.rank = current_rank().rank
        self.world_size = WORLD_SIZE

    @endpoint
    async def store_tensor(self):
        t = torch.tensor([self.rank])
        await ts.put(f"{self.rank}_tensor", t)

    @endpoint
    async def print_tensor(self):
        other_rank = (self.rank + 1) % self.world_size
        t = await ts.get(f"{other_rank}_tensor")
        print(f"Rank=[{self.rank}] Fetched {t} from {other_rank=}")


async def main():

    # Create a store instance
    await ts.initialize()

    actors = await spawn_actors(WORLD_SIZE, ExampleActor, "example_actors")

    # Calls "store_tensor" on each actor instance
    await actors.store_tensor.call()
    await actors.print_tensor.call()

if  __name__ == "__main__":
    asyncio.run(main())

# Expected output
# [0] [2] Rank=[2] Fetched tensor([3]) from other_rank=3
# [0] [0] Rank=[0] Fetched tensor([1]) from other_rank=1
# [0] [3] Rank=[3] Fetched tensor([0]) from other_rank=0
# [0] [1] Rank=[1] Fetched tensor([2]) from other_rank=2
