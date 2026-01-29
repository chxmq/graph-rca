# Foursquare Incident (2010-10-04)
Root Cause: MongoDB ran out of memory due to a query pattern with low locality (fetching full history for every check-in), leading to catastrophic failure.
Category: Database/Memory
Source: https://web.archive.org/web/20230602082218/https://news.ycombinator.com/item?id=1769761
