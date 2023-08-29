import face_swap
import time
import os
from azure.storage.queue import QueueClient
from azure.storage.blob import BlobServiceClient, BlobClient
import logging
import json

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s [%(levelname)s]: %(message)s",  # Set the log format
    datefmt="%Y-%m-%d %H:%M:%S"  # Set the date format
)


def listen_to_azure_queue():
    connection_string = os.environ.get("AZ_STORAGE_CONN")
    queue_name = os.environ.get("FACE_SWAP_QUEUE")
    source_container = os.environ.get("SOURCE_CONTAINER")
    target_container = os.environ.get("TARGET_CONTAINER")
    result_container = os.environ.get("RESULT_CONTAINER")

    queue_client = QueueClient.from_connection_string(connection_string, queue_name)

    while True:
        # Receive one message from the queue
        message = queue_client.receive_message()

        if message:
            start = time.time()
            logging.info("Received message:", message.content)

            json_data = json.loads(message.content)

            logging.info(json_data)
            source_img = json_data["source_image"]
            target_img = json_data["target_image"]
            result_img = json_data["result_image"]

            logging.info(f"Getting source file {source_img} from {source_container}")
            download_blob(connection_string, source_container, source_img, f'./temp/{source_img}')

            logging.info(f"Getting target file {target_img} from {target_container}")
            download_blob(connection_string, target_container, target_img, f'./temp/{target_img}')

            logging.info("Performing face swap")
            face_swap.process_image(f'./temp/{source_img}', f'./temp/{target_img}', f"./temp/{result_img}")

            logging.info(f"Uploading resulting image {result_img} to {result_container}")
            upload_blob(connection_string, result_container, result_img, f"./temp/{result_img}")

            queue_client.delete_message(message)

            end = time.time()
            logging.info(f"Total time taken {end - start}")
        else:
            # Introduce a delay before the next polling
            logging.info("Nothing in queue")
            time.sleep(5)


def download_blob(connection_string, container_name, blob_name, destination_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open(destination_path, "wb") as file:
        blob_data = blob_client.download_blob()
        file.write(blob_data.readall())


def upload_blob(connection_string, container_name, blob_name, source_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    logging.info(f"Uploading resulting image {source_path} {container_name} {blob_name}")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    logging.info(f"Uploading resulting image {source_path}")

    with open(source_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)


if __name__ == "__main__":
    listen_to_azure_queue()
