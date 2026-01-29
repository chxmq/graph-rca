# NPM Outage Jan 28, 2014
Root Cause: Varnish restart behavior.
The `restart` command in VCL reset `req.backend` to the top-most defined backend.
