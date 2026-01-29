from collect_incidents import IncidentDataCollector
import json
import os

def populate_batch_6():
    collector = IncidentDataCollector()
    
    # 41. Google - URL Blacklist (2009)
    print("Populating Google Blacklist 2009...")
    collector.create_incident_directory("041")
    collector.save_metadata("041", {
        "company": "Google",
        "category": "Software/Logic",
        "date": "2009-01-31",
        "severity": "Medium"
    })
    collector.save_ground_truth("041", {
        "incident_id": "041",
        "source": "https://googleblog.blogspot.com/2009/01/this-site-may-harm-your-computer-on.html",
        "root_cause": "The '/' character was accidentally checked into the malicious URL blacklist, causing every site on the internet to be flagged as harmful.",
        "root_cause_timestamp": "2009-01-31T00:00:00Z",
        "category": "Software",
        "severity": "Medium",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 42. Google - Provisioning Deletion (2023)
    print("Populating Google Provisioning 2023...")
    collector.create_incident_directory("042")
    collector.save_metadata("042", {
        "company": "Google",
        "category": "Infrastructure/Config",
        "date": "2023-01-01",
        "severity": "High"
    })
    collector.save_ground_truth("042", {
        "incident_id": "042",
        "source": "https://cloud.google.com/blog/products/infrastructure/details-of-google-cloud-gcve-incident",
        "root_cause": "GCVE provisioning used a legacy option that created a 'fixed-term' contract, leading to automatic deletion of the environment upon expiry.",
        "root_cause_timestamp": "2023-01-01T00:00:00Z",
        "category": "Infrastructure",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 43. TUI - Passenger Weight Bug (2021)
    print("Populating TUI 2021...")
    collector.create_incident_directory("043")
    collector.save_metadata("043", {
        "company": "TUI",
        "category": "Software/Logic",
        "date": "2020-07-21",
        "severity": "Critical"
    })
    collector.save_ground_truth("043", {
        "incident_id": "043",
        "source": "https://assets.publishing.service.gov.uk/media/604f423be90e077fdf88493f/Boeing_737-8K5_G-TAWG_04-21.pdf",
        "root_cause": "System upgrade caused female passengers with the title 'Miss' to be misidentified as children, using 35kg instead of 69kg for takeoff mass calculations.",
        "root_cause_timestamp": "2020-07-21T00:00:00Z",
        "category": "Software",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 44. BBC Online - DB Overload (2014)
    print("Populating BBC 2014...")
    collector.create_incident_directory("044")
    collector.save_metadata("044", {
        "company": "BBC",
        "category": "Database/Scaling",
        "date": "2014-07-01",
        "severity": "High"
    })
    collector.save_ground_truth("044", {
        "incident_id": "044",
        "source": "https://www.bbc.co.uk/blogs/internet/entries/a37b0470-47d4-3991-82bb-a7d5b8803771",
        "root_cause": "Database backend overload caused throttling; services without local caching timed out and failed completely.",
        "root_cause_timestamp": "2014-07-01T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 45. Buildkite - DB Downgrade (2018)
    print("Populating Buildkite 2018...")
    collector.create_incident_directory("045")
    collector.save_metadata("045", {
        "company": "Buildkite",
        "category": "Database/Scaling",
        "date": "2018-08-23",
        "severity": "High"
    })
    collector.save_ground_truth("045", {
        "incident_id": "045",
        "source": "https://building.buildkite.com/outage-post-mortem-for-august-23rd-82b619a3679b",
        "root_cause": "Database capacity downgrade (cost saving) resulted in insufficient resources for peak load, causing cascading collapse.",
        "root_cause_timestamp": "2018-08-23T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 46. NASA - Mars Pathfinder Priority Inversion (1997)
    print("Populating NASA 1997...")
    collector.create_incident_directory("046")
    collector.save_metadata("046", {
        "company": "NASA",
        "category": "Software/OS",
        "date": "1997-07-04",
        "severity": "Medium"
    })
    collector.save_ground_truth("046", {
        "incident_id": "046",
        "source": "https://en.wikipedia.org/wiki/Priority_inversion",
        "root_cause": "Priority inversion in VxWorks OS where a low-priority task held a mutex needed by a high-priority task, while a medium-priority task blocked the low-priority one.",
        "root_cause_timestamp": "1997-07-04T00:00:00Z",
        "category": "Software",
        "severity": "Medium",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 47. OpenAI - Redis Corruption (2023)
    print("Populating OpenAI 2023...")
    collector.create_incident_directory("047")
    collector.save_metadata("047", {
        "company": "OpenAI",
        "category": "Database/Redis",
        "date": "2023-03-20",
        "severity": "High"
    })
    collector.save_ground_truth("047", {
        "incident_id": "047",
        "source": "https://openai.com/blog/march-20-chatgpt-outage",
        "root_cause": "Redis request/response queues became corrupted and out-of-sequence, revealing user data to other users.",
        "root_cause_timestamp": "2023-03-20T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 48. Grab - Predictive Autoscaling (2024)
    print("Populating Grab 2024...")
    collector.create_incident_directory("048")
    collector.save_metadata("048", {
        "company": "Grab",
        "category": "Infrastructure/Autoscaling",
        "date": "2024-01-15",
        "severity": "Medium"
    })
    collector.save_ground_truth("048", {
        "incident_id": "048",
        "source": "https://sreweekly.com/sre-weekly-issue-500/",
        "root_cause": "Predictive autoscaling algorithm miscalculated the required capacity for a surge in demand, leading to resource exhaustion.",
        "root_cause_timestamp": "2024-01-15T00:00:00Z",
        "category": "Infrastructure",
        "severity": "Medium",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 49. Microsoft Azure - Cooling Failure (2022)
    print("Populating Azure 2022...")
    collector.create_incident_directory("049")
    collector.save_metadata("049", {
        "company": "Microsoft",
        "category": "Infrastructure/Power",
        "date": "2022-07-19",
        "severity": "High"
    })
    collector.save_ground_truth("049", {
        "incident_id": "049",
        "source": "https://status.cloud.google.com/incidents/fmEL9i2fArADKawkZAa2",
        "root_cause": "Simultaneous failure of multiple redundant cooling systems in a data center during a heatwave forced a shutdown of servers.",
        "root_cause_timestamp": "2022-07-19T06:33:00Z",
        "category": "Infrastructure",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 50. Atlassian - Data Deletion (2022)
    print("Populating Atlassian 2022...")
    collector.create_incident_directory("050")
    collector.save_metadata("050", {
        "company": "Atlassian",
        "category": "Software/Script",
        "date": "2022-04-05",
        "severity": "Critical"
    })
    collector.save_ground_truth("050", {
        "incident_id": "050",
        "source": "https://www.atlassian.com/engineering/post-incident-review-april-2022-outage",
        "root_cause": "Faulty script intended to decommission a legacy app inadvertently targeted active customer environments for deletion.",
        "root_cause_timestamp": "2022-04-05T07:38:00Z",
        "category": "Software",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

if __name__ == "__main__":
    populate_batch_6()
    print("\n✅ Successfully populated Batch 6!")
