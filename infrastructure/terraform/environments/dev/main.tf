locals { name = "credit-risk-${var.environment}" }

resource "aws_kms_key" "main" { description = "Credit-risk platform encryption"; enable_key_rotation = true }
resource "aws_s3_bucket" "artifacts" { bucket = var.artifact_bucket_name }
resource "aws_s3_bucket_public_access_block" "artifacts" { bucket = aws_s3_bucket.artifacts.id; block_public_acls=true; block_public_policy=true; ignore_public_acls=true; restrict_public_buckets=true }
resource "aws_s3_bucket_versioning" "artifacts" { bucket=aws_s3_bucket.artifacts.id; versioning_configuration { status="Enabled" } }
resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" { bucket=aws_s3_bucket.artifacts.id; rule { apply_server_side_encryption_by_default { kms_master_key_id=aws_kms_key.main.arn; sse_algorithm="aws:kms" } } }

resource "aws_dynamodb_table" "decisions" { name="${local.name}-decisions"; billing_mode="PAY_PER_REQUEST"; hash_key="application_id"; attribute { name="application_id"; type="S" }; server_side_encryption { enabled=true; kms_key_arn=aws_kms_key.main.arn }; point_in_time_recovery { enabled=true } }

data "aws_iam_policy_document" "lambda_assume" { statement { actions=["sts:AssumeRole"]; principals { type="Service"; identifiers=["lambda.amazonaws.com"] } } }
resource "aws_iam_role" "lambda" { name="${local.name}-lambda"; assume_role_policy=data.aws_iam_policy_document.lambda_assume.json }
resource "aws_iam_role_policy" "lambda" { role=aws_iam_role.lambda.id; policy=jsonencode({Version="2012-10-17",Statement=[{Effect="Allow",Action=["dynamodb:GetItem","dynamodb:PutItem"],Resource=aws_dynamodb_table.decisions.arn},{Effect="Allow",Action=["sagemaker:InvokeEndpoint"],Resource="arn:aws:sagemaker:${var.aws_region}:*:endpoint/${local.name}-*"},{Effect="Allow",Action=["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],Resource="*"}]}) }

resource "aws_lambda_function" "api" { filename=var.lambda_zip; function_name="${local.name}-api"; role=aws_iam_role.lambda.arn; handler="lambda_handler.handler"; runtime="python3.12"; timeout=30; source_code_hash=fileexists(var.lambda_zip) ? filebase64sha256(var.lambda_zip) : null; environment { variables={DECISION_TABLE=aws_dynamodb_table.decisions.name,LOG_LEVEL="INFO"} } }
resource "aws_apigatewayv2_api" "api" { name=local.name; protocol_type="HTTP" }
resource "aws_apigatewayv2_integration" "lambda" { api_id=aws_apigatewayv2_api.api.id; integration_type="AWS_PROXY"; integration_uri=aws_lambda_function.api.invoke_arn; payload_format_version="2.0" }
resource "aws_apigatewayv2_route" "default" { api_id=aws_apigatewayv2_api.api.id; route_key="$default"; target="integrations/${aws_apigatewayv2_integration.lambda.id}" }
resource "aws_apigatewayv2_stage" "default" { api_id=aws_apigatewayv2_api.api.id; name="$default"; auto_deploy=true }
resource "aws_lambda_permission" "api" { statement_id="AllowApiGateway"; action="lambda:InvokeFunction"; function_name=aws_lambda_function.api.function_name; principal="apigateway.amazonaws.com"; source_arn="${aws_apigatewayv2_api.api.execution_arn}/*/*" }

resource "aws_sfn_state_machine" "governance" { name="${local.name}-governance"; role_arn=aws_iam_role.sfn.arn; definition=jsonencode({StartAt="ValidateMaturedCohort",States={ValidateMaturedCohort={Type="Pass",Next="BuildFeatures"},BuildFeatures={Type="Pass",Next="TrainAndCalibrate"},TrainAndCalibrate={Type="Pass",Next="GovernanceGate"},GovernanceGate={Type="Pass",Next="RegisterCandidate"},RegisterCandidate={Type="Succeed"}}}) }
data "aws_iam_policy_document" "sfn_assume" { statement { actions=["sts:AssumeRole"]; principals { type="Service"; identifiers=["states.amazonaws.com"] } } }
resource "aws_iam_role" "sfn" { name="${local.name}-sfn"; assume_role_policy=data.aws_iam_policy_document.sfn_assume.json }
resource "aws_cloudwatch_log_group" "api" { name="/aws/lambda/${local.name}-api"; retention_in_days=30; kms_key_id=aws_kms_key.main.arn }

