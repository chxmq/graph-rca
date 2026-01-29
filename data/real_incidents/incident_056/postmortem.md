# Strava Incident (2014-07-29)
Root Cause: Signed 32-bit integer limit hit on a primary key in the uploads database, causing all new uploads to fail.
Category: Database/Software
Source: https://engineering.strava.com/the-upload-outage-of-july-29-2014/
