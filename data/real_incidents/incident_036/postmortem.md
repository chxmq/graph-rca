# Amazon Incident (2020-11-25)
Root Cause: Scaling the Kinesis front-end fleet exceeded OS thread limits due to memory-intensive configuration, breaking Cognito/Lambda.
Category: Infrastructure/OS Config
Source: https://aws.amazon.com/message/11201/
