from collect_incidents import IncidentDataCollector
import json
import os

def populate_batch_7():
    collector = IncidentDataCollector()
    
    # 51. Heroku - Git Push Broken (2021)
    print("Populating Heroku Git 2021...")
    collector.create_incident_directory("051")
    collector.save_metadata("051", {
        "company": "Heroku",
        "category": "Software/Deploy",
        "date": "2021-01-01",
        "severity": "Medium"
    })
    collector.save_ground_truth("051", {
        "incident_id": "051",
        "source": "https://blog.heroku.com/how-i-broke-git-push-heroku-main",
        "root_cause": "Incorrect deployment process caused new configuration variables not to be used by the code that required them, breaking git push.",
        "root_cause_timestamp": "2021-01-01T00:00:00Z",
        "category": "Software",
        "severity": "Medium",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 52. Azure - Storage Config (2014)
    print("Populating Azure Storage 2014...")
    collector.create_incident_directory("052")
    collector.save_metadata("052", {
        "company": "Microsoft",
        "category": "Configuration/Storage",
        "date": "2014-11-18",
        "severity": "Critical"
    })
    collector.save_ground_truth("052", {
        "incident_id": "052",
        "source": "https://azure.microsoft.com/en-us/blog/update-on-azure-storage-service-interruption/",
        "root_cause": "A configuration change intended to improve performance across the storage service was deployed incorrectly, taking down Azure Storage globally.",
        "root_cause_timestamp": "2014-11-18T00:00:00Z",
        "category": "Configuration",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 53. Google - BGP Announcement (2016)
    print("Populating Google BGP 2016...")
    collector.create_incident_directory("053")
    collector.save_metadata("053", {
        "company": "Google",
        "category": "Network/BGP",
        "date": "2016-04-11",
        "severity": "High"
    })
    collector.save_ground_truth("053", {
        "incident_id": "053",
        "source": "https://status.cloud.google.com/incident/compute/16007",
        "root_cause": "An automated configuration generation tool removed all Google Compute Engine IP blocks from BGP announcements, leading to massive reachability issues.",
        "root_cause_timestamp": "2016-04-11T00:00:00Z",
        "category": "Network",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 54. Google - Europe Load Balancing (2021)
    print("Populating Google Europe LB 2021...")
    collector.create_incident_directory("054")
    collector.save_metadata("054", {
        "company": "Google",
        "category": "Infrastructure/Load Balancing",
        "date": "2021-08-24",
        "severity": "High"
    })
    collector.save_ground_truth("054", {
        "incident_id": "054",
        "source": "https://status.cloud.google.com/incidents/1xkAB1KmLrh5g3v9ZEZ7",
        "root_cause": "A new infrastructure feature triggered a latent issue within the internal network load balancer code in Europe regions.",
        "root_cause_timestamp": "2021-08-24T00:00:00Z",
        "category": "Infrastructure",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 55. Reddit - Kubernetes Upgrade (2023)
    print("Populating Reddit K8s 2023...")
    collector.create_incident_directory("055")
    collector.save_metadata("055", {
        "company": "Reddit",
        "category": "Infrastructure/Kubernetes",
        "date": "2023-03-14",
        "severity": "Critical"
    })
    collector.save_ground_truth("055", {
        "incident_id": "055",
        "source": "https://www.reddit.com/r/RedditEng/comments/11xx5o0/you_broke_reddit_the_piday_outage/",
        "root_cause": "Critical Kubernetes cluster upgrade failed because node metadata changed between versions, breaking workload networking for over 5 hours.",
        "root_cause_timestamp": "2023-03-14T00:00:00Z",
        "category": "Infrastructure",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 56. Strava - Integer Limit (2014)
    print("Populating Strava 2014...")
    collector.create_incident_directory("056")
    collector.save_metadata("056", {
        "company": "Strava",
        "category": "Database/Software",
        "date": "2014-07-29",
        "severity": "Medium"
    })
    collector.save_ground_truth("056", {
        "incident_id": "056",
        "source": "https://engineering.strava.com/the-upload-outage-of-july-29-2014/",
        "root_cause": "Signed 32-bit integer limit hit on a primary key in the uploads database, causing all new uploads to fail.",
        "root_cause_timestamp": "2014-07-29T00:00:00Z",
        "category": "Database",
        "severity": "Medium",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 57. Stripe - Manual Operation (2015)
    print("Populating Stripe 2015...")
    collector.create_incident_directory("057")
    collector.save_metadata("057", {
        "company": "Stripe",
        "category": "Database/Human Error",
        "date": "2015-10-08",
        "severity": "High"
    })
    collector.save_ground_truth("057", {
        "incident_id": "057",
        "source": "https://support.stripe.com/questions/outage-postmortem-2015-10-08-utc",
        "root_cause": "Incorrectly executed manual database operation (missing dependency) caused the Stripe API to go down for 90 minutes.",
        "root_cause_timestamp": "2015-10-08T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 58. Heroku - Pinned Packages (2018)
    print("Populating Heroku Pinned 2018...")
    collector.create_incident_directory("058")
    collector.save_metadata("058", {
        "company": "Heroku",
        "category": "Software/OS",
        "date": "2018-05-01",
        "severity": "Medium"
    })
    collector.save_ground_truth("058", {
        "incident_id": "058",
        "source": "https://status.heroku.com/incidents/1042",
        "root_cause": "An upstream 'apt' update broke pinned packages, causing write permission failures to /dev on customer dynos.",
        "root_cause_timestamp": "2018-05-01T00:00:00Z",
        "category": "Software",
        "severity": "Medium",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 59. Amazon S3 - Message Corruption (2008)
    print("Populating Amazon S3 2008...")
    collector.create_incident_directory("059")
    collector.save_metadata("059", {
        "company": "Amazon",
        "category": "Software/Gossip",
        "date": "2008-07-20",
        "severity": "High"
    })
    collector.save_ground_truth("059", {
        "incident_id": "059",
        "source": "https://status.aws.amazon.com/s3-20080720.html",
        "root_cause": "Single corrupted bit in a gossip message caused distributed server state to overwhelm the S3 request processing fleet.",
        "root_cause_timestamp": "2008-07-20T00:00:00Z",
        "category": "Software",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 60. WhatsApp - Server Config (2021)
    print("Populating WhatsApp 2021...")
    collector.create_incident_directory("060")
    collector.save_metadata("060", {
        "company": "WhatsApp",
        "category": "Network/Config",
        "date": "2021-10-04",
        "severity": "Critical"
    })
    collector.save_ground_truth("060", {
        "incident_id": "060",
        "source": "https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/",
        "root_cause": "Configuration changes to Facebook's backbone routers also took down WhatsApp and Instagram globally.",
        "root_cause_timestamp": "2021-10-04T15:00:00Z",
        "category": "Network",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

if __name__ == "__main__":
    populate_batch_7()
    print("\n✅ Successfully populated Batch 7!")
