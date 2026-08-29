"""Thin AWS adapters; boto3 is imported only when an AWS adapter is instantiated."""

from __future__ import annotations

import json


class SageMakerRiskClient:
    def __init__(self, endpoint_name: str, client=None):
        if client is None:
            import boto3

            client = boto3.client("sagemaker-runtime")
        self.client = client
        self.endpoint_name = endpoint_name

    def score(self, features: dict) -> dict:
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(features).encode(),
        )
        return json.loads(response["Body"].read())


class DynamoDecisionStore:
    def __init__(self, table_name: str, resource=None):
        if resource is None:
            import boto3

            resource = boto3.resource("dynamodb")
        self.table = resource.Table(table_name)

    def put(self, record: dict):
        self.table.put_item(Item=record)

    def get(self, application_id: str):
        return self.table.get_item(Key={"application_id": application_id}).get("Item")
