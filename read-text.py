from dotenv import load_dotenv
import os
import time
from PIL import Image
from flask import Flask, render_template

# Import namespaces for Azure Computer Vision
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

app = Flask(__name__)

# Load environment variables
load_dotenv()
cog_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
cog_key = os.getenv('COG_SERVICE_KEY')

# Authenticate Azure AI Vision client
credential = CognitiveServicesCredentials(cog_key)
cv_client = ComputerVisionClient(cog_endpoint, credential)


@app.route("/read-text")
def read_text():
    image_file = os.path.join('images', 'image3.jpg')
    output = GetTextRead(image_file)
    return render_template("read_text.html", output=output)

    

def GetTextRead(image_file):
    print('Reading text in {}\n'.format(image_file))
    # Use Read API to read text in image
    with open(image_file, mode="rb") as image_data:
        read_op = cv_client.read_in_stream(image_data, raw=True)

    # Get the async operation ID so we can check for the results
    operation_location = read_op.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    # Wait for the asynchronous operation to complete
    while True:
        read_results = cv_client.get_read_result(operation_id)
        if read_results.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
            break
        time.sleep(1)

    # If the operation was successful, process the text line by line
    result_text = []
    if read_results.status == OperationStatusCodes.succeeded:
        for page in read_results.analyze_result.read_results:
            for line in page.lines:
                result_text.append(line.text)

    return result_text


if __name__ == "__main__":
    app.run(debug=True)
