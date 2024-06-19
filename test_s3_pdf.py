from io import BytesIO

import boto3
from PyPDF2 import PdfReader


s3 = boto3.client("s3")
obj = s3.get_object(Bucket="myapi-private-bucket", Key="databaserag-krishna/mature/308c6bd8-632f-40da-8519-5c6eda2a542c.pdf")
reader = PdfReader(BytesIO(obj["Body"].read()))
for page in reader.pages:
    print(page)
