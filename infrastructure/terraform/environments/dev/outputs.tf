output "api_url" { value=aws_apigatewayv2_api.api.api_endpoint }
output "artifact_bucket" { value=aws_s3_bucket.artifacts.id }
output "decision_table" { value=aws_dynamodb_table.decisions.name }

