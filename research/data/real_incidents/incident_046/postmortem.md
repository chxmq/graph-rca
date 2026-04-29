# NASA Incident (1997-07-04)
Root Cause: Priority inversion in VxWorks OS where a low-priority task held a mutex needed by a high-priority task, while a medium-priority task blocked the low-priority one.
Category: Software/OS
Source: https://en.wikipedia.org/wiki/Priority_inversion
