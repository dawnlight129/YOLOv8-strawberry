
from torch.utils.tensorboard import SummaryWriter
# import tensorboard
# from packaging.version import Version
# from .writer import FileWriter, SummaryWriter
writer = SummaryWriter("logs")

# writer.add_image()
for i in range(100):

    writer.add_scalar("y=2x",2*i, i)

writer.close()

