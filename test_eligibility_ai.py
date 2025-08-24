import json
import sys
import os

# Add the parent directory to sys.path to import from eligibility_checker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set a dummy GROQ API key for testing (you should set the real one in .env file)
os.environ["GROQ_API_KEY"] = "test_key"

from eligibility.eligibility_checker import check_detailed_eligibility

# Test payload from the user
test_payload = {
    "user": {
        "id": "68a9f0783c5872f5cc01",
        "name": "Soumyaraj Bag",
        "course": "B.Tech",
        "stream": "Computer Science Engineering", 
        "batch": "2026",
        "institute": "Sample Institute",
        "avg_cgpa": 4.66,
        "activeBacklogs": 0,
        "skillsCount": 2,
        "skills": [
            {"name": "Computer Science", "level": "Advanced"},
            {"name": "Programming", "level": "Intermediate"}
        ]
    },
    "post": {
        "postId": "68aa44ad003a15d9810c",
        "title": "Campus Drive by Deloitte India for the 2026 passing out batch B.Tech (All)",
        "type": "Opportunity",
        "criteria": {
            "cgpa": 6.5,
            "backlogs": 0,
            "skills": [
                "CS",
                "IT", 
                "ECE",
                "EE"
            ],
            "courses": [],
            "experience": ""
        },
        "eligibility": {
            "minCGPA": "6.5",
            "branches": [
                "All"
            ],
            "batch": [
                "2026"
            ]
        }
    }
}

print("Testing Eligibility Checker with Groq AI Integration...")
print("=" * 60)

try:
    result = check_detailed_eligibility(test_payload)
    print("ELIGIBILITY CHECK RESULT:")
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"User: {result['name']}")
    print(f"CGPA: {result['avg_cgpa']}")
    print(f"Eligible: {result['isEligible']}")
    print(f"Key Issues:")
    for key, value in result['breakdown'].items():
        if value['status'] == 'fail':
            print(f"  - {key.upper()}: {value['message']}")
            
except Exception as e:
    print(f"Error testing eligibility checker: {e}")
    import traceback
    traceback.print_exc()
