import modal

stub = modal.App("oasis-resnet")
volume = modal.Volume.from_name("oasis-data", create_if_missing=True)

@stub.function(volumes={"/data": volume})
def train():
    import os
    print(os.listdir("/data"))  # Access your uploaded dataset
    ...
