# Security, AWS, and scale

Use private buckets with block-public-access, versioning, KMS encryption, TLS, scoped IAM roles, Secrets Manager/Parameter Store, VPC endpoints, private subnets where appropriate, CloudTrail, short retention for sanitized logs, and explicit deletion/retention controls. Never hardcode keys or log raw bureau payloads.

For millions of applications, partition S3 Parquet by product/origination year/month, use SageMaker Processing or distributed feature jobs, Batch Transform for portfolios, autoscaled multi-AZ endpoints, model registry aliases, cached versioned policies, DynamoDB idempotency keys, and decoupled monitoring aggregation. Extend product-by-product; do not reuse personal-loan EAD or calibration blindly.

