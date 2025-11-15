import modal
import pathlib

app = modal.App("oasis-resnet")
vol = modal.Volume.from_name("oasis-storage", environment_name="brain-age")

VOL_MOUNT_PATH = pathlib.Path("/data")

@app.function(volumes={str(VOL_MOUNT_PATH): vol})
def my_function():
    file_path = VOL_MOUNT_PATH /"data"/ "example_2.txt"
    with open(file_path, "w") as f:
        f.write("Hello, Modal Volumes!")
    print(f"Wrote to {file_path}")

    vol.commit()
