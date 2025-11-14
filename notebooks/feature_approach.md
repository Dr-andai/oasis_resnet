# Using both imaging and clinical (categorical) features together to predict a continuous outcome (age)
Option A: Early Fusion
You merge both inputs before the model starts learning:
Treat the diagnosis (0/1) as an additional “channel” or feature map
e.g. concatenate diagnosis to each image voxel (broadcasted) or to an embedding vector derived from the image.

Option B: Late Fusion
Let the CNN process MRI images → output a latent embedding (say 512-d vector).
Then concatenate that embedding with the diagnosis and pass both into an MLP head that predicts age.