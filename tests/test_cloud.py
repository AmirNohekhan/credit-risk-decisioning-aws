import io
import json

from credit_risk.cloud import DynamoDecisionStore, SageMakerRiskClient


class Runtime:
    def invoke_endpoint(self, **kwargs):
        return {"Body": io.BytesIO(json.dumps({"pd_12m": 0.1}).encode())}


class Table:
    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        self.items[Item["application_id"]] = Item

    def get_item(self, Key):
        return {"Item": self.items[Key["application_id"]]}


class Resource:
    def __init__(self):
        self.table = Table()

    def Table(self, name):
        return self.table


def test_aws_adapters_without_credentials():
    assert SageMakerRiskClient("endpoint", Runtime()).score({"x": 1})["pd_12m"] == 0.1
    store = DynamoDecisionStore("table", Resource())
    store.put({"application_id": "a", "decision": "APPROVE"})
    assert store.get("a")["decision"] == "APPROVE"
